//===- Decl.cpp - Declaration AST Node Implementation ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the Decl subclasses.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/Decl.h"
#include "Linkage.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/ASTDiagnostic.h"
#include "clang/AST/ASTLambda.h"
#include "clang/AST/ASTMutationListener.h"
#include "clang/AST/Attr.h"
#include "clang/AST/CanonicalType.h"
#include "clang/AST/DeclBase.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/DeclOpenMP.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/DeclarationName.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/ExternalASTSource.h"
#include "clang/AST/ODRHash.h"
#include "clang/AST/PrettyDeclStackTrace.h"
#include "clang/AST/PrettyPrinter.h"
#include "clang/AST/Randstruct.h"
#include "clang/AST/RecordLayout.h"
#include "clang/AST/Redeclarable.h"
#include "clang/AST/Stmt.h"
#include "clang/AST/TemplateBase.h"
#include "clang/AST/Type.h"
#include "clang/AST/TypeLoc.h"
#include "clang/Basic/Builtins.h"
#include "clang/Basic/IdentifierTable.h"
#include "clang/Basic/LLVM.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Basic/Linkage.h"
#include "clang/Basic/Module.h"
#include "clang/Basic/NoSanitizeList.h"
#include "clang/Basic/PartialDiagnostic.h"
#include "clang/Basic/Sanitizers.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/Specifiers.h"
#include "clang/Basic/TargetCXXABI.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Basic/Visibility.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Triple.h"
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <type_traits>

using namespace clang;

Decl *clang::getPrimaryMergedDecl(Decl *D) {
  return D->getASTContext().getPrimaryMergedDecl(D);
}

void PrettyDeclStackTraceEntry::print(raw_ostream &OS) const {
  SourceLocation Loc = this->Loc;
  if (!Loc.isValid() && TheDecl) Loc = TheDecl->getLocation();
  if (Loc.isValid()) {
    Loc.print(OS, Context.getSourceManager());
    OS << ": ";
  }
  OS << Message;

  if (auto *ND = dyn_cast_if_present<NamedDecl>(TheDecl)) {
    OS << " '";
    ND->getNameForDiagnostic(OS, Context.getPrintingPolicy(), true);
    OS << "'";
  }

  OS << '\n';
}

// Defined here so that it can be inlined into its direct callers.
bool Decl::isOutOfLine() const {
  return !getLexicalDeclContext()->Equals(getDeclContext());
}

TranslationUnitDecl::TranslationUnitDecl(ASTContext &ctx)
    : Decl(TranslationUnit, nullptr, SourceLocation()),
      DeclContext(TranslationUnit), redeclarable_base(ctx), Ctx(ctx) {}

//===----------------------------------------------------------------------===//
// NamedDecl Implementation
//===----------------------------------------------------------------------===//

// Visibility rules aren't rigorously externally specified, but here
// are the basic principles behind what we implement:
//
// 1. An explicit visibility attribute is generally a direct expression
// of the user's intent and should be honored.  Only the innermost
// visibility attribute applies.  If no visibility attribute applies,
// global visibility settings are considered.
//
// 2. There is one caveat to the above: on or in a template pattern,
// an explicit visibility attribute is just a default rule, and
// visibility can be decreased by the visibility of template
// arguments.  But this, too, has an exception: an attribute on an
// explicit specialization or instantiation causes all the visibility
// restrictions of the template arguments to be ignored.
//
// 3. A variable that does not otherwise have explicit visibility can
// be restricted by the visibility of its type.
//
// 4. A visibility restriction is explicit if it comes from an
// attribute (or something like it), not a global visibility setting.
// When emitting a reference to an external symbol, visibility
// restrictions are ignored unless they are explicit.
//
// 5. When computing the visibility of a non-type, including a
// non-type member of a class, only non-type visibility restrictions
// are considered: the 'visibility' attribute, global value-visibility
// settings, and a few special cases like __private_extern.
//
// 6. When computing the visibility of a type, including a type member
// of a class, only type visibility restrictions are considered:
// the 'type_visibility' attribute and global type-visibility settings.
// However, a 'visibility' attribute counts as a 'type_visibility'
// attribute on any declaration that only has the former.
//
// The visibility of a "secondary" entity, like a template argument,
// is computed using the kind of that entity, not the kind of the
// primary entity for which we are computing visibility.  For example,
// the visibility of a specialization of either of these templates:
//   template <class T, bool (&compare)(T, X)> bool has_match(list<T>, X);
//   template <class T, bool (&compare)(T, X)> class matcher;
// is restricted according to the type visibility of the argument 'T',
// the type visibility of 'bool(&)(T,X)', and the value visibility of
// the argument function 'compare'.  That 'has_match' is a value
// and 'matcher' is a type only matters when looking for attributes
// and settings from the immediate context.

/// Does this computation kind permit us to consider additional
/// visibility settings from attributes and the like?
static bool hasExplicitVisibilityAlready(LVComputationKind computation) {
  return computation.IgnoreExplicitVisibility;
}

/// Given an LVComputationKind, return one of the same type/value sort
/// that records that it already has explicit visibility.
static LVComputationKind
withExplicitVisibilityAlready(LVComputationKind Kind) {
  Kind.IgnoreExplicitVisibility = true;
  return Kind;
}

static std::optional<Visibility> getExplicitVisibility(const NamedDecl *D,
                                                       LVComputationKind kind) {
  assert(!kind.IgnoreExplicitVisibility &&
         "asking for explicit visibility when we shouldn't be");
  return D->getExplicitVisibility(kind.getExplicitVisibilityKind());
}

/// Is the given declaration a "type" or a "value" for the purposes of
/// visibility computation?
static bool usesTypeVisibility(const NamedDecl *D) {
  return isa<TypeDecl>(D) ||
         isa<ClassTemplateDecl>(D) ||
         isa<ObjCInterfaceDecl>(D);
}

/// Does the given declaration have member specialization information,
/// and if so, is it an explicit specialization?
template <class T>
static std::enable_if_t<!std::is_base_of_v<RedeclarableTemplateDecl, T>, bool>
isExplicitMemberSpecialization(const T *D) {
  if (const MemberSpecializationInfo *member =
        D->getMemberSpecializationInfo()) {
    return member->isExplicitSpecialization();
  }
  return false;
}

/// For templates, this question is easier: a member template can't be
/// explicitly instantiated, so there's a single bit indicating whether
/// or not this is an explicit member specialization.
static bool isExplicitMemberSpecialization(const RedeclarableTemplateDecl *D) {
  return D->isMemberSpecialization();
}

/// Given a visibility attribute, return the explicit visibility
/// associated with it.
template <class T>
static Visibility getVisibilityFromAttr(const T *attr) {
  switch (attr->getVisibility()) {
  case T::Default:
    return DefaultVisibility;
  case T::Hidden:
    return HiddenVisibility;
  case T::Protected:
    return ProtectedVisibility;
  }
  llvm_unreachable("bad visibility kind");
}

/// Return the explicit visibility of the given declaration.
static std::optional<Visibility>
getVisibilityOf(const NamedDecl *D, NamedDecl::ExplicitVisibilityKind kind) {
  // If we're ultimately computing the visibility of a type, look for
  // a 'type_visibility' attribute before looking for 'visibility'.
  if (kind == NamedDecl::VisibilityForType) {
    if (const auto *A = D->getAttr<TypeVisibilityAttr>()) {
      return getVisibilityFromAttr(A);
    }
  }

  // If this declaration has an explicit visibility attribute, use it.
  if (const auto *A = D->getAttr<VisibilityAttr>()) {
    return getVisibilityFromAttr(A);
  }

  return std::nullopt;
}

LinkageInfo LinkageComputer::getLVForType(const Type &T,
                                          LVComputationKind computation) {
  if (computation.IgnoreAllVisibility)
    return LinkageInfo(T.getLinkage(), DefaultVisibility, true);
  return getTypeLinkageAndVisibility(&T);
}

/// Get the most restrictive linkage for the types in the given
/// template parameter list.  For visibility purposes, template
/// parameters are part of the signature of a template.
LinkageInfo LinkageComputer::getLVForTemplateParameterList(
    const TemplateParameterList *Params, LVComputationKind computation) {
  LinkageInfo LV;
  for (const NamedDecl *P : *Params) {
    // Template type parameters are the most common and never
    // contribute to visibility, pack or not.
    if (isa<TemplateTypeParmDecl>(P))
      continue;

    // Non-type template parameters can be restricted by the value type, e.g.
    //   template <enum X> class A { ... };
    // We have to be careful here, though, because we can be dealing with
    // dependent types.
    if (const auto *NTTP = dyn_cast<NonTypeTemplateParmDecl>(P)) {
      // Handle the non-pack case first.
      if (!NTTP->isExpandedParameterPack()) {
        if (!NTTP->getType()->isDependentType()) {
          LV.merge(getLVForType(*NTTP->getType(), computation));
        }
        continue;
      }

      // Look at all the types in an expanded pack.
      for (unsigned i = 0, n = NTTP->getNumExpansionTypes(); i != n; ++i) {
        QualType type = NTTP->getExpansionType(i);
        if (!type->isDependentType())
          LV.merge(getTypeLinkageAndVisibility(type));
      }
      continue;
    }

    // Template template parameters can be restricted by their
    // template parameters, recursively.
    const auto *TTP = cast<TemplateTemplateParmDecl>(P);

    // Handle the non-pack case first.
    if (!TTP->isExpandedParameterPack()) {
      LV.merge(getLVForTemplateParameterList(TTP->getTemplateParameters(),
                                             computation));
      continue;
    }

    // Look at all expansions in an expanded pack.
    for (unsigned i = 0, n = TTP->getNumExpansionTemplateParameters();
           i != n; ++i) {
      LV.merge(getLVForTemplateParameterList(
          TTP->getExpansionTemplateParameters(i), computation));
    }
  }

  return LV;
}

static const Decl *getOutermostFuncOrBlockContext(const Decl *D) {
  const Decl *Ret = nullptr;
  const DeclContext *DC = D->getDeclContext();
  while (DC->getDeclKind() != Decl::TranslationUnit) {
    if (isa<FunctionDecl>(DC) || isa<BlockDecl>(DC))
      Ret = cast<Decl>(DC);
    DC = DC->getParent();
  }
  return Ret;
}

/// Get the most restrictive linkage for the types and
/// declarations in the given template argument list.
///
/// Note that we don't take an LVComputationKind because we always
/// want to honor the visibility of template arguments in the same way.
LinkageInfo
LinkageComputer::getLVForTemplateArgumentList(ArrayRef<TemplateArgument> Args,
                                              LVComputationKind computation) {
  LinkageInfo LV;

  for (const TemplateArgument &Arg : Args) {
    switch (Arg.getKind()) {
    case TemplateArgument::Null:
    case TemplateArgument::Integral:
    case TemplateArgument::Expression:
      continue;

    case TemplateArgument::Type:
      LV.merge(getLVForType(*Arg.getAsType(), computation));
      continue;

    case TemplateArgument::Declaration: {
      const NamedDecl *ND = Arg.getAsDecl();
      assert(!usesTypeVisibility(ND));
      LV.merge(getLVForDecl(ND, computation));
      continue;
    }

    case TemplateArgument::NullPtr:
      LV.merge(getTypeLinkageAndVisibility(Arg.getNullPtrType()));
      continue;

    case TemplateArgument::StructuralValue:
      LV.merge(getLVForValue(Arg.getAsStructuralValue(), computation));
      continue;

    case TemplateArgument::Template:
    case TemplateArgument::TemplateExpansion:
      if (TemplateDecl *Template =
              Arg.getAsTemplateOrTemplatePattern().getAsTemplateDecl())
        LV.merge(getLVForDecl(Template, computation));
      continue;

    case TemplateArgument::Pack:
      LV.merge(getLVForTemplateArgumentList(Arg.getPackAsArray(), computation));
      continue;
    }
    llvm_unreachable("bad template argument kind");
  }

  return LV;
}

LinkageInfo
LinkageComputer::getLVForTemplateArgumentList(const TemplateArgumentList &TArgs,
                                              LVComputationKind computation) {
  return getLVForTemplateArgumentList(TArgs.asArray(), computation);
}

static bool shouldConsiderTemplateVisibility(const FunctionDecl *fn,
                        const FunctionTemplateSpecializationInfo *specInfo) {
  // Include visibility from the template parameters and arguments
  // only if this is not an explicit instantiation or specialization
  // with direct explicit visibility.  (Implicit instantiations won't
  // have a direct attribute.)
  if (!specInfo->isExplicitInstantiationOrSpecialization())
    return true;

  return !fn->hasAttr<VisibilityAttr>();
}

/// Merge in template-related linkage and visibility for the given
/// function template specialization.
///
/// We don't need a computation kind here because we can assume
/// LVForValue.
///
/// \param[out] LV the computation to use for the parent
void LinkageComputer::mergeTemplateLV(
    LinkageInfo &LV, const FunctionDecl *fn,
    const FunctionTemplateSpecializationInfo *specInfo,
    LVComputationKind computation) {
  bool considerVisibility =
    shouldConsiderTemplateVisibility(fn, specInfo);

  FunctionTemplateDecl *temp = specInfo->getTemplate();
  // Merge information from the template declaration.
  LinkageInfo tempLV = getLVForDecl(temp, computation);
  // The linkage of the specialization should be consistent with the
  // template declaration.
  LV.setLinkage(tempLV.getLinkage());

  // Merge information from the template parameters.
  LinkageInfo paramsLV =
      getLVForTemplateParameterList(temp->getTemplateParameters(), computation);
  LV.mergeMaybeWithVisibility(paramsLV, considerVisibility);

  // Merge information from the template arguments.
  const TemplateArgumentList &templateArgs = *specInfo->TemplateArguments;
  LinkageInfo argsLV = getLVForTemplateArgumentList(templateArgs, computation);
  LV.mergeMaybeWithVisibility(argsLV, considerVisibility);
}

/// Does the given declaration have a direct visibility attribute
/// that would match the given rules?
static bool hasDirectVisibilityAttribute(const NamedDecl *D,
                                         LVComputationKind computation) {
  if (computation.IgnoreAllVisibility)
    return false;

  return (computation.isTypeVisibility() && D->hasAttr<TypeVisibilityAttr>()) ||
         D->hasAttr<VisibilityAttr>();
}

/// Should we consider visibility associated with the template
/// arguments and parameters of the given class template specialization?
static bool shouldConsiderTemplateVisibility(
                                 const ClassTemplateSpecializationDecl *spec,
                                 LVComputationKind computation) {
  // Include visibility from the template parameters and arguments
  // only if this is not an explicit instantiation or specialization
  // with direct explicit visibility (and note that implicit
  // instantiations won't have a direct attribute).
  //
  // Furthermore, we want to ignore template parameters and arguments
  // for an explicit specialization when computing the visibility of a
  // member thereof with explicit visibility.
  //
  // This is a bit complex; let's unpack it.
  //
  // An explicit class specialization is an independent, top-level
  // declaration.  As such, if it or any of its members has an
  // explicit visibility attribute, that must directly express the
  // user's intent, and we should honor it.  The same logic applies to
  // an explicit instantiation of a member of such a thing.

  // Fast path: if this is not an explicit instantiation or
  // specialization, we always want to consider template-related
  // visibility restrictions.
  if (!spec->isExplicitInstantiationOrSpecialization())
    return true;

  // This is the 'member thereof' check.
  if (spec->isExplicitSpecialization() &&
      hasExplicitVisibilityAlready(computation))
    return false;

  return !hasDirectVisibilityAttribute(spec, computation);
}

/// Merge in template-related linkage and visibility for the given
/// class template specialization.
void LinkageComputer::mergeTemplateLV(
    LinkageInfo &LV, const ClassTemplateSpecializationDecl *spec,
    LVComputationKind computation) {
  bool considerVisibility = shouldConsiderTemplateVisibility(spec, computation);

  // Merge information from the template parameters, but ignore
  // visibility if we're only considering template arguments.
  ClassTemplateDecl *temp = spec->getSpecializedTemplate();
  // Merge information from the template declaration.
  LinkageInfo tempLV = getLVForDecl(temp, computation);
  // The linkage of the specialization should be consistent with the
  // template declaration.
  LV.setLinkage(tempLV.getLinkage());

  LinkageInfo paramsLV =
    getLVForTemplateParameterList(temp->getTemplateParameters(), computation);
  LV.mergeMaybeWithVisibility(paramsLV,
           considerVisibility && !hasExplicitVisibilityAlready(computation));

  // Merge information from the template arguments.  We ignore
  // template-argument visibility if we've got an explicit
  // instantiation with a visibility attribute.
  const TemplateArgumentList &templateArgs = spec->getTemplateArgs();
  LinkageInfo argsLV = getLVForTemplateArgumentList(templateArgs, computation);
  if (considerVisibility)
    LV.mergeVisibility(argsLV);
  LV.mergeExternalVisibility(argsLV);
}

/// Should we consider visibility associated with the template
/// arguments and parameters of the given variable template
/// specialization? As usual, follow class template specialization
/// logic up to initialization.
static bool shouldConsiderTemplateVisibility(
                                 const VarTemplateSpecializationDecl *spec,
                                 LVComputationKind computation) {
  // Include visibility from the template parameters and arguments
  // only if this is not an explicit instantiation or specialization
  // with direct explicit visibility (and note that implicit
  // instantiations won't have a direct attribute).
  if (!spec->isExplicitInstantiationOrSpecialization())
    return true;

  // An explicit variable specialization is an independent, top-level
  // declaration.  As such, if it has an explicit visibility attribute,
  // that must directly express the user's intent, and we should honor
  // it.
  if (spec->isExplicitSpecialization() &&
      hasExplicitVisibilityAlready(computation))
    return false;

  return !hasDirectVisibilityAttribute(spec, computation);
}

/// Merge in template-related linkage and visibility for the given
/// variable template specialization. As usual, follow class template
/// specialization logic up to initialization.
void LinkageComputer::mergeTemplateLV(LinkageInfo &LV,
                                      const VarTemplateSpecializationDecl *spec,
                                      LVComputationKind computation) {
  bool considerVisibility = shouldConsiderTemplateVisibility(spec, computation);

  // Merge information from the template parameters, but ignore
  // visibility if we're only considering template arguments.
  VarTemplateDecl *temp = spec->getSpecializedTemplate();
  LinkageInfo tempLV =
    getLVForTemplateParameterList(temp->getTemplateParameters(), computation);
  LV.mergeMaybeWithVisibility(tempLV,
           considerVisibility && !hasExplicitVisibilityAlready(computation));

  // Merge information from the template arguments.  We ignore
  // template-argument visibility if we've got an explicit
  // instantiation with a visibility attribute.
  const TemplateArgumentList &templateArgs = spec->getTemplateArgs();
  LinkageInfo argsLV = getLVForTemplateArgumentList(templateArgs, computation);
  if (considerVisibility)
    LV.mergeVisibility(argsLV);
  LV.mergeExternalVisibility(argsLV);
}

static bool useInlineVisibilityHidden(const NamedDecl *D) {
  // FIXME: we should warn if -fvisibility-inlines-hidden is used with c.
  const LangOptions &Opts = D->getASTContext().getLangOpts();
  if (!Opts.CPlusPlus || !Opts.InlineVisibilityHidden)
    return false;

  const auto *FD = dyn_cast<FunctionDecl>(D);
  if (!FD)
    return false;

  TemplateSpecializationKind TSK = TSK_Undeclared;
  if (FunctionTemplateSpecializationInfo *spec
      = FD->getTemplateSpecializationInfo()) {
    TSK = spec->getTemplateSpecializationKind();
  } else if (MemberSpecializationInfo *MSI =
             FD->getMemberSpecializationInfo()) {
    TSK = MSI->getTemplateSpecializationKind();
  }

  const FunctionDecl *Def = nullptr;
  // InlineVisibilityHidden only applies to definitions, and
  // isInlined() only gives meaningful answers on definitions
  // anyway.
  return TSK != TSK_ExplicitInstantiationDeclaration &&
    TSK != TSK_ExplicitInstantiationDefinition &&
    FD->hasBody(Def) && Def->isInlined() && !Def->hasAttr<GNUInlineAttr>();
}

template <typename T> static bool isFirstInExternCContext(T *D) {
  const T *First = D->getFirstDecl();
  return First->isInExternCContext();
}

static bool isSingleLineLanguageLinkage(const Decl &D) {
  if (const auto *SD = dyn_cast<LinkageSpecDecl>(D.getDeclContext()))
    if (!SD->hasBraces())
      return true;
  return false;
}

static bool isDeclaredInModuleInterfaceOrPartition(const NamedDecl *D) {
  if (auto *M = D->getOwningModule())
    return M->isInterfaceOrPartition();
  return false;
}

static LinkageInfo getExternalLinkageFor(const NamedDecl *D) {
  return LinkageInfo::external();
}

static StorageClass getStorageClass(const Decl *D) {
  if (auto *TD = dyn_cast<TemplateDecl>(D))
    D = TD->getTemplatedDecl();
  if (D) {
    if (auto *VD = dyn_cast<VarDecl>(D))
      return VD->getStorageClass();
    if (auto *FD = dyn_cast<FunctionDecl>(D))
      return FD->getStorageClass();
  }
  return SC_None;
}

LinkageInfo
LinkageComputer::getLVForNamespaceScopeDecl(const NamedDecl *D,
                                            LVComputationKind computation,
                                            bool IgnoreVarTypeLinkage) {
  assert(D->getDeclContext()->getRedeclContext()->isFileContext() &&
         "Not a name having namespace scope");
  ASTContext &Context = D->getASTContext();

  // C++ [basic.link]p3:
  //   A name having namespace scope (3.3.6) has internal linkage if it
  //   is the name of

  if (getStorageClass(D->getCanonicalDecl()) == SC_Static) {
    // - a variable, variable template, function, or function template
    //   that is explicitly declared static; or
    // (This bullet corresponds to C99 6.2.2p3.)
    return LinkageInfo::internal();
  }

  if (const auto *Var = dyn_cast<VarDecl>(D)) {
    // - a non-template variable of non-volatile const-qualified type, unless
    //   - it is explicitly declared extern, or
    //   - it is declared in the purview of a module interface unit
    //     (outside the private-module-fragment, if any) or module partition, or
    //   - it is inline, or
    //   - it was previously declared and the prior declaration did not have
    //     internal linkage
    // (There is no equivalent in C99.)
    if (Context.getLangOpts().CPlusPlus && Var->getType().isConstQualified() &&
        !Var->getType().isVolatileQualified() && !Var->isInline() &&
        !isDeclaredInModuleInterfaceOrPartition(Var) &&
        !isa<VarTemplateSpecializationDecl>(Var) &&
        !Var->getDescribedVarTemplate()) {
      const VarDecl *PrevVar = Var->getPreviousDecl();
      if (PrevVar)
        return getLVForDecl(PrevVar, computation);

      if (Var->getStorageClass() != SC_Extern &&
          Var->getStorageClass() != SC_PrivateExtern &&
          !isSingleLineLanguageLinkage(*Var))
        return LinkageInfo::internal();
    }

    for (const VarDecl *PrevVar = Var->getPreviousDecl(); PrevVar;
         PrevVar = PrevVar->getPreviousDecl()) {
      if (PrevVar->getStorageClass() == SC_PrivateExtern &&
          Var->getStorageClass() == SC_None)
        return getDeclLinkageAndVisibility(PrevVar);
      // Explicitly declared static.
      if (PrevVar->getStorageClass() == SC_Static)
        return LinkageInfo::internal();
    }
  } else if (const auto *IFD = dyn_cast<IndirectFieldDecl>(D)) {
    //   - a data member of an anonymous union.
    const VarDecl *VD = IFD->getVarDecl();
    assert(VD && "Expected a VarDecl in this IndirectFieldDecl!");
    return getLVForNamespaceScopeDecl(VD, computation, IgnoreVarTypeLinkage);
  }
  assert(!isa<FieldDecl>(D) && "Didn't expect a FieldDecl!");

  // FIXME: This gives internal linkage to names that should have no linkage
  // (those not covered by [basic.link]p6).
  if (D->isInAnonymousNamespace()) {
    const auto *Var = dyn_cast<VarDecl>(D);
    const auto *Func = dyn_cast<FunctionDecl>(D);
    // FIXME: The check for extern "C" here is not justified by the standard
    // wording, but we retain it from the pre-DR1113 model to avoid breaking
    // code.
    //
    // C++11 [basic.link]p4:
    //   An unnamed namespace or a namespace declared directly or indirectly
    //   within an unnamed namespace has internal linkage.
    if ((!Var || !isFirstInExternCContext(Var)) &&
        (!Func || !isFirstInExternCContext(Func)))
      return LinkageInfo::internal();
  }

  // Set up the defaults.

  // C99 6.2.2p5:
  //   If the declaration of an identifier for an object has file
  //   scope and no storage-class specifier, its linkage is
  //   external.
  LinkageInfo LV = getExternalLinkageFor(D);

  if (!hasExplicitVisibilityAlready(computation)) {
    if (std::optional<Visibility> Vis = getExplicitVisibility(D, computation)) {
      LV.mergeVisibility(*Vis, true);
    } else {
      // If we're declared in a namespace with a visibility attribute,
      // use that namespace's visibility, and it still counts as explicit.
      for (const DeclContext *DC = D->getDeclContext();
           !isa<TranslationUnitDecl>(DC);
           DC = DC->getParent()) {
        const auto *ND = dyn_cast<NamespaceDecl>(DC);
        if (!ND) continue;
        if (std::optional<Visibility> Vis =
                getExplicitVisibility(ND, computation)) {
          LV.mergeVisibility(*Vis, true);
          break;
        }
      }
    }

    // Add in global settings if the above didn't give us direct visibility.
    if (!LV.isVisibilityExplicit()) {
      // Use global type/value visibility as appropriate.
      Visibility globalVisibility =
          computation.isValueVisibility()
              ? Context.getLangOpts().getValueVisibilityMode()
              : Context.getLangOpts().getTypeVisibilityMode();
      LV.mergeVisibility(globalVisibility, /*explicit*/ false);

      // If we're paying attention to global visibility, apply
      // -finline-visibility-hidden if this is an inline method.
      if (useInlineVisibilityHidden(D))
        LV.mergeVisibility(HiddenVisibility, /*visibilityExplicit=*/false);
    }
  }

  // C++ [basic.link]p4:

  //   A name having namespace scope that has not been given internal linkage
  //   above and that is the name of
  //   [...bullets...]
  //   has its linkage determined as follows:
  //     - if the enclosing namespace has internal linkage, the name has
  //       internal linkage; [handled above]
  //     - otherwise, if the declaration of the name is attached to a named
  //       module and is not exported, the name has module linkage;
  //     - otherwise, the name has external linkage.
  // LV is currently set up to handle the last two bullets.
  //
  //   The bullets are:

  //     - a variable; or
  if (const auto *Var = dyn_cast<VarDecl>(D)) {
    // GCC applies the following optimization to variables and static
    // data members, but not to functions:
    //
    // Modify the variable's LV by the LV of its type unless this is
    // C or extern "C".  This follows from [basic.link]p9:
    //   A type without linkage shall not be used as the type of a
    //   variable or function with external linkage unless
    //    - the entity has C language linkage, or
    //    - the entity is declared within an unnamed namespace, or
    //    - the entity is not used or is defined in the same
    //      translation unit.
    // and [basic.link]p10:
    //   ...the types specified by all declarations referring to a
    //   given variable or function shall be identical...
    // C does not have an equivalent rule.
    //
    // Ignore this if we've got an explicit attribute;  the user
    // probably knows what they're doing.
    //
    // Note that we don't want to make the variable non-external
    // because of this, but unique-external linkage suits us.

    if (Context.getLangOpts().CPlusPlus && !isFirstInExternCContext(Var) &&
        !IgnoreVarTypeLinkage) {
      LinkageInfo TypeLV = getLVForType(*Var->getType(), computation);
      if (!isExternallyVisible(TypeLV.getLinkage()))
        return LinkageInfo::uniqueExternal();
      if (!LV.isVisibilityExplicit())
        LV.mergeVisibility(TypeLV);
    }

    if (Var->getStorageClass() == SC_PrivateExtern)
      LV.mergeVisibility(HiddenVisibility, true);

    // Note that Sema::MergeVarDecl already takes care of implementing
    // C99 6.2.2p4 and propagating the visibility attribute, so we don't have
    // to do it here.

    // As per function and class template specializations (below),
    // consider LV for the template and template arguments.  We're at file
    // scope, so we do not need to worry about nested specializations.
    if (const auto *spec = dyn_cast<VarTemplateSpecializationDecl>(Var)) {
      mergeTemplateLV(LV, spec, computation);
    }

  //     - a function; or
  } else if (const auto *Function = dyn_cast<FunctionDecl>(D)) {
    // In theory, we can modify the function's LV by the LV of its
    // type unless it has C linkage (see comment above about variables
    // for justification).  In practice, GCC doesn't do this, so it's
    // just too painful to make work.

    if (Function->getStorageClass() == SC_PrivateExtern)
      LV.mergeVisibility(HiddenVisibility, true);

    // OpenMP target declare device functions are not callable from the host so
    // they should not be exported from the device image. This applies to all
    // functions as the host-callable kernel functions are emitted at codegen.
    if (Context.getLangOpts().OpenMP &&
        Context.getLangOpts().OpenMPIsTargetDevice &&
        ((Context.getTargetInfo().getTriple().isAMDGPU() ||
          Context.getTargetInfo().getTriple().isNVPTX()) ||
         OMPDeclareTargetDeclAttr::isDeclareTargetDeclaration(Function)))
      LV.mergeVisibility(HiddenVisibility, /*newExplicit=*/false);

    // Note that Sema::MergeCompatibleFunctionDecls already takes care of
    // merging storage classes and visibility attributes, so we don't have to
    // look at previous decls in here.

    // In C++, then if the type of the function uses a type with
    // unique-external linkage, it's not legally usable from outside
    // this translation unit.  However, we should use the C linkage
    // rules instead for extern "C" declarations.
    if (Context.getLangOpts().CPlusPlus && !isFirstInExternCContext(Function)) {
      // Only look at the type-as-written. Otherwise, deducing the return type
      // of a function could change its linkage.
      QualType TypeAsWritten = Function->getType();
      if (TypeSourceInfo *TSI = Function->getTypeSourceInfo())
        TypeAsWritten = TSI->getType();
      if (!isExternallyVisible(TypeAsWritten->getLinkage()))
        return LinkageInfo::uniqueExternal();
    }

    // Consider LV from the template and the template arguments.
    // We're at file scope, so we do not need to worry about nested
    // specializations.
    if (FunctionTemplateSpecializationInfo *specInfo
                               = Function->getTemplateSpecializationInfo()) {
      mergeTemplateLV(LV, Function, specInfo, computation);
    }

  //     - a named class (Clause 9), or an unnamed class defined in a
  //       typedef declaration in which the class has the typedef name
  //       for linkage purposes (7.1.3); or
  //     - a named enumeration (7.2), or an unnamed enumeration
  //       defined in a typedef declaration in which the enumeration
  //       has the typedef name for linkage purposes (7.1.3); or
  } else if (const auto *Tag = dyn_cast<TagDecl>(D)) {
    // Unnamed tags have no linkage.
    if (!Tag->hasNameForLinkage())
      return LinkageInfo::none();

    // If this is a class template specialization, consider the
    // linkage of the template and template arguments.  We're at file
    // scope, so we do not need to worry about nested specializations.
    if (const auto *spec = dyn_cast<ClassTemplateSpecializationDecl>(Tag)) {
      mergeTemplateLV(LV, spec, computation);
    }

  // FIXME: This is not part of the C++ standard any more.
  //     - an enumerator belonging to an enumeration with external linkage; or
  } else if (isa<EnumConstantDecl>(D)) {
    LinkageInfo EnumLV = getLVForDecl(cast<NamedDecl>(D->getDeclContext()),
                                      computation);
    if (!isExternalFormalLinkage(EnumLV.getLinkage()))
      return LinkageInfo::none();
    LV.merge(EnumLV);

  //     - a template
  } else if (const auto *temp = dyn_cast<TemplateDecl>(D)) {
    bool considerVisibility = !hasExplicitVisibilityAlready(computation);
    LinkageInfo tempLV =
      getLVForTemplateParameterList(temp->getTemplateParameters(), computation);
    LV.mergeMaybeWithVisibility(tempLV, considerVisibility);

  //     An unnamed namespace or a namespace declared directly or indirectly
  //     within an unnamed namespace has internal linkage. All other namespaces
  //     have external linkage.
  //
  // We handled names in anonymous namespaces above.
  } else if (isa<NamespaceDecl>(D)) {
    return LV;

  // By extension, we assign external linkage to Objective-C
  // interfaces.
  } else if (isa<ObjCInterfaceDecl>(D)) {
    // fallout

  } else if (auto *TD = dyn_cast<TypedefNameDecl>(D)) {
    // A typedef declaration has linkage if it gives a type a name for
    // linkage purposes.
    if (!TD->getAnonDeclWithTypedefName(/*AnyRedecl*/true))
      return LinkageInfo::none();

  } else if (isa<MSGuidDecl>(D)) {
    // A GUID behaves like an inline variable with external linkage. Fall
    // through.

  // Everything not covered here has no linkage.
  } else {
    return LinkageInfo::none();
  }

  // If we ended up with non-externally-visible linkage, visibility should
  // always be default.
  if (!isExternallyVisible(LV.getLinkage()))
    return LinkageInfo(LV.getLinkage(), DefaultVisibility, false);

  return LV;
}

LinkageInfo
LinkageComputer::getLVForClassMember(const NamedDecl *D,
                                     LVComputationKind computation,
                                     bool IgnoreVarTypeLinkage) {
  // Only certain class members have linkage.  Note that fields don't
  // really have linkage, but it's convenient to say they do for the
  // purposes of calculating linkage of pointer-to-data-member
  // template arguments.
  //
  // Templates also don't officially have linkage, but since we ignore
  // the C++ standard and look at template arguments when determining
  // linkage and visibility of a template specialization, we might hit
  // a template template argument that way. If we do, we need to
  // consider its linkage.
  if (!(isa<CXXMethodDecl>(D) ||
        isa<VarDecl>(D) ||
        isa<FieldDecl>(D) ||
        isa<IndirectFieldDecl>(D) ||
        isa<TagDecl>(D) ||
        isa<TemplateDecl>(D)))
    return LinkageInfo::none();

  LinkageInfo LV;

  // If we have an explicit visibility attribute, merge that in.
  if (!hasExplicitVisibilityAlready(computation)) {
    if (std::optional<Visibility> Vis = getExplicitVisibility(D, computation))
      LV.mergeVisibility(*Vis, true);
    // If we're paying attention to global visibility, apply
    // -finline-visibility-hidden if this is an inline method.
    //
    // Note that we do this before merging information about
    // the class visibility.
    if (!LV.isVisibilityExplicit() && useInlineVisibilityHidden(D))
      LV.mergeVisibility(HiddenVisibility, /*visibilityExplicit=*/false);
  }

  // If this class member has an explicit visibility attribute, the only
  // thing that can change its visibility is the template arguments, so
  // only look for them when processing the class.
  LVComputationKind classComputation = computation;
  if (LV.isVisibilityExplicit())
    classComputation = withExplicitVisibilityAlready(computation);

  LinkageInfo classLV =
    getLVForDecl(cast<RecordDecl>(D->getDeclContext()), classComputation);
  // The member has the same linkage as the class. If that's not externally
  // visible, we don't need to compute anything about the linkage.
  // FIXME: If we're only computing linkage, can we bail out here?
  if (!isExternallyVisible(classLV.getLinkage()))
    return classLV;


  // Otherwise, don't merge in classLV yet, because in certain cases
  // we need to completely ignore the visibility from it.

  // Specifically, if this decl exists and has an explicit attribute.
  const NamedDecl *explicitSpecSuppressor = nullptr;

  if (const auto *MD = dyn_cast<CXXMethodDecl>(D)) {
    // Only look at the type-as-written. Otherwise, deducing the return type
    // of a function could change its linkage.
    QualType TypeAsWritten = MD->getType();
    if (TypeSourceInfo *TSI = MD->getTypeSourceInfo())
      TypeAsWritten = TSI->getType();
    if (!isExternallyVisible(TypeAsWritten->getLinkage()))
      return LinkageInfo::uniqueExternal();

    // If this is a method template specialization, use the linkage for
    // the template parameters and arguments.
    if (FunctionTemplateSpecializationInfo *spec
           = MD->getTemplateSpecializationInfo()) {
      mergeTemplateLV(LV, MD, spec, computation);
      if (spec->isExplicitSpecialization()) {
        explicitSpecSuppressor = MD;
      } else if (isExplicitMemberSpecialization(spec->getTemplate())) {
        explicitSpecSuppressor = spec->getTemplate()->getTemplatedDecl();
      }
    } else if (isExplicitMemberSpecialization(MD)) {
      explicitSpecSuppressor = MD;
    }

    // OpenMP target declare device functions are not callable from the host so
    // they should not be exported from the device image. This applies to all
    // functions as the host-callable kernel functions are emitted at codegen.
    ASTContext &Context = D->getASTContext();
    if (Context.getLangOpts().OpenMP &&
        Context.getLangOpts().OpenMPIsTargetDevice &&
        ((Context.getTargetInfo().getTriple().isAMDGPU() ||
          Context.getTargetInfo().getTriple().isNVPTX()) ||
         OMPDeclareTargetDeclAttr::isDeclareTargetDeclaration(MD)))
      LV.mergeVisibility(HiddenVisibility, /*newExplicit=*/false);

  } else if (const auto *RD = dyn_cast<CXXRecordDecl>(D)) {
    if (const auto *spec = dyn_cast<ClassTemplateSpecializationDecl>(RD)) {
      mergeTemplateLV(LV, spec, computation);
      if (spec->isExplicitSpecialization()) {
        explicitSpecSuppressor = spec;
      } else {
        const ClassTemplateDecl *temp = spec->getSpecializedTemplate();
        if (isExplicitMemberSpecialization(temp)) {
          explicitSpecSuppressor = temp->getTemplatedDecl();
        }
      }
    } else if (isExplicitMemberSpecialization(RD)) {
      explicitSpecSuppressor = RD;
    }

  // Static data members.
  } else if (const auto *VD = dyn_cast<VarDecl>(D)) {
    if (const auto *spec = dyn_cast<VarTemplateSpecializationDecl>(VD))
      mergeTemplateLV(LV, spec, computation);

    // Modify the variable's linkage by its type, but ignore the
    // type's visibility unless it's a definition.
    if (!IgnoreVarTypeLinkage) {
      LinkageInfo typeLV = getLVForType(*VD->getType(), computation);
      // FIXME: If the type's linkage is not externally visible, we can
      // give this static data member UniqueExternalLinkage.
      if (!LV.isVisibilityExplicit() && !classLV.isVisibilityExplicit())
        LV.mergeVisibility(typeLV);
      LV.mergeExternalVisibility(typeLV);
    }

    if (isExplicitMemberSpecialization(VD)) {
      explicitSpecSuppressor = VD;
    }

  // Template members.
  } else if (const auto *temp = dyn_cast<TemplateDecl>(D)) {
    bool considerVisibility =
      (!LV.isVisibilityExplicit() &&
       !classLV.isVisibilityExplicit() &&
       !hasExplicitVisibilityAlready(computation));
    LinkageInfo tempLV =
      getLVForTemplateParameterList(temp->getTemplateParameters(), computation);
    LV.mergeMaybeWithVisibility(tempLV, considerVisibility);

    if (const auto *redeclTemp = dyn_cast<RedeclarableTemplateDecl>(temp)) {
      if (isExplicitMemberSpecialization(redeclTemp)) {
        explicitSpecSuppressor = temp->getTemplatedDecl();
      }
    }
  }

  // We should never be looking for an attribute directly on a template.
  assert(!explicitSpecSuppressor || !isa<TemplateDecl>(explicitSpecSuppressor));

  // If this member is an explicit member specialization, and it has
  // an explicit attribute, ignore visibility from the parent.
  bool considerClassVisibility = true;
  if (explicitSpecSuppressor &&
      // optimization: hasDVA() is true only with explicit visibility.
      LV.isVisibilityExplicit() &&
      classLV.getVisibility() != DefaultVisibility &&
      hasDirectVisibilityAttribute(explicitSpecSuppressor, computation)) {
    considerClassVisibility = false;
  }

  // Finally, merge in information from the class.
  LV.mergeMaybeWithVisibility(classLV, considerClassVisibility);
  return LV;
}

void NamedDecl::anchor() {}

bool NamedDecl::isLinkageValid() const {
  if (!hasCachedLinkage())
    return true;

  Linkage L = LinkageComputer{}
                  .computeLVForDecl(this, LVComputationKind::forLinkageOnly())
                  .getLinkage();
  return L == getCachedLinkage();
}

bool NamedDecl::isPlaceholderVar(const LangOptions &LangOpts) const {
  // [C++2c] [basic.scope.scope]/p5
  // A declaration is name-independent if its name is _ and it declares
  // - a variable with automatic storage duration,
  // - a structured binding not inhabiting a namespace scope,
  // - the variable introduced by an init-capture
  // - or a non-static data member.

  if (!LangOpts.CPlusPlus || !getIdentifier() ||
      !getIdentifier()->isPlaceholder())
    return false;
  if (isa<FieldDecl>(this))
    return true;
  if (const auto *IFD = dyn_cast<IndirectFieldDecl>(this)) {
    if (!getDeclContext()->isFunctionOrMethod() &&
        !getDeclContext()->isRecord())
      return false;
    const VarDecl *VD = IFD->getVarDecl();
    return !VD || VD->getStorageDuration() == SD_Automatic;
  }
  // and it declares a variable with automatic storage duration
  if (const auto *VD = dyn_cast<VarDecl>(this)) {
    if (isa<ParmVarDecl>(VD))
      return false;
    if (VD->isInitCapture())
      return true;
    return VD->getStorageDuration() == StorageDuration::SD_Automatic;
  }
  if (const auto *BD = dyn_cast<BindingDecl>(this);
      BD && getDeclContext()->isFunctionOrMethod()) {
    const VarDecl *VD = BD->getHoldingVar();
    return !VD || VD->getStorageDuration() == StorageDuration::SD_Automatic;
  }
  return false;
}

ReservedIdentifierStatus
NamedDecl::isReserved(const LangOptions &LangOpts) const {
  const IdentifierInfo *II = getIdentifier();

  // This triggers at least for CXXLiteralIdentifiers, which we already checked
  // at lexing time.
  if (!II)
    return ReservedIdentifierStatus::NotReserved;

  ReservedIdentifierStatus Status = II->isReserved(LangOpts);
  if (isReservedAtGlobalScope(Status) && !isReservedInAllContexts(Status)) {
    // This name is only reserved at global scope. Check if this declaration
    // conflicts with a global scope declaration.
    if (isa<ParmVarDecl>(this) || isTemplateParameter())
      return ReservedIdentifierStatus::NotReserved;

    // C++ [dcl.link]/7:
    //   Two declarations [conflict] if [...] one declares a function or
    //   variable with C language linkage, and the other declares [...] a
    //   variable that belongs to the global scope.
    //
    // Therefore names that are reserved at global scope are also reserved as
    // names of variables and functions with C language linkage.
    const DeclContext *DC = getDeclContext()->getRedeclContext();
    if (DC->isTranslationUnit())
      return Status;
    if (auto *VD = dyn_cast<VarDecl>(this))
      if (VD->isExternC())
        return ReservedIdentifierStatus::StartsWithUnderscoreAndIsExternC;
    if (auto *FD = dyn_cast<FunctionDecl>(this))
      if (FD->isExternC())
        return ReservedIdentifierStatus::StartsWithUnderscoreAndIsExternC;
    return ReservedIdentifierStatus::NotReserved;
  }

  return Status;
}

ObjCStringFormatFamily NamedDecl::getObjCFStringFormattingFamily() const {
  StringRef name = getName();
  if (name.empty()) return SFF_None;

  if (name.front() == 'C')
    if (name == "CFStringCreateWithFormat" ||
        name == "CFStringCreateWithFormatAndArguments" ||
        name == "CFStringAppendFormat" ||
        name == "CFStringAppendFormatAndArguments")
      return SFF_CFString;
  return SFF_None;
}

Linkage NamedDecl::getLinkageInternal() const {
  // We don't care about visibility here, so ask for the cheapest
  // possible visibility analysis.
  return LinkageComputer{}
      .getLVForDecl(this, LVComputationKind::forLinkageOnly())
      .getLinkage();
}

/// Determine whether D is attached to a named module.
static bool isInNamedModule(const NamedDecl *D) {
  if (auto *M = D->getOwningModule())
    return M->isNamedModule();
  return false;
}

static bool isExportedFromModuleInterfaceUnit(const NamedDecl *D) {
  // FIXME: Handle isModulePrivate.
  switch (D->getModuleOwnershipKind()) {
  case Decl::ModuleOwnershipKind::Unowned:
  case Decl::ModuleOwnershipKind::ReachableWhenImported:
  case Decl::ModuleOwnershipKind::ModulePrivate:
    return false;
  case Decl::ModuleOwnershipKind::Visible:
  case Decl::ModuleOwnershipKind::VisibleWhenImported:
    return isInNamedModule(D);
  }
  llvm_unreachable("unexpected module ownership kind");
}

/// Get the linkage from a semantic point of view. Entities in
/// anonymous namespaces are external (in c++98).
Linkage NamedDecl::getFormalLinkage() const {
  Linkage InternalLinkage = getLinkageInternal();

  // C++ [basic.link]p4.8:
  //   - if the declaration of the name is attached to a named module and is not
  //   exported
  //     the name has module linkage;
  //
  // [basic.namespace.general]/p2
  //   A namespace is never attached to a named module and never has a name with
  //   module linkage.
  if (isInNamedModule(this) && InternalLinkage == Linkage::External &&
      !isExportedFromModuleInterfaceUnit(
          cast<NamedDecl>(this->getCanonicalDecl())) &&
      !isa<NamespaceDecl>(this))
    InternalLinkage = Linkage::Module;

  return clang::getFormalLinkage(InternalLinkage);
}

LinkageInfo NamedDecl::getLinkageAndVisibility() const {
  return LinkageComputer{}.getDeclLinkageAndVisibility(this);
}

static std::optional<Visibility>
getExplicitVisibilityAux(const NamedDecl *ND,
                         NamedDecl::ExplicitVisibilityKind kind,
                         bool IsMostRecent) {
  assert(!IsMostRecent || ND == ND->getMostRecentDecl());

  // Check the declaration itself first.
  if (std::optional<Visibility> V = getVisibilityOf(ND, kind))
    return V;

  // If this is a member class of a specialization of a class template
  // and the corresponding decl has explicit visibility, use that.
  if (const auto *RD = dyn_cast<CXXRecordDecl>(ND)) {
    CXXRecordDecl *InstantiatedFrom = RD->getInstantiatedFromMemberClass();
    if (InstantiatedFrom)
      return getVisibilityOf(InstantiatedFrom, kind);
  }

  // If there wasn't explicit visibility there, and this is a
  // specialization of a class template, check for visibility
  // on the pattern.
  if (const auto *spec = dyn_cast<ClassTemplateSpecializationDecl>(ND)) {
    // Walk all the template decl till this point to see if there are
    // explicit visibility attributes.
    const auto *TD = spec->getSpecializedTemplate()->getTemplatedDecl();
    while (TD != nullptr) {
      auto Vis = getVisibilityOf(TD, kind);
      if (Vis != std::nullopt)
        return Vis;
      TD = TD->getPreviousDecl();
    }
    return std::nullopt;
  }

  // Use the most recent declaration.
  if (!IsMostRecent && !isa<NamespaceDecl>(ND)) {
    const NamedDecl *MostRecent = ND->getMostRecentDecl();
    if (MostRecent != ND)
      return getExplicitVisibilityAux(MostRecent, kind, true);
  }

  if (const auto *Var = dyn_cast<VarDecl>(ND)) {
    if (Var->isStaticDataMember()) {
      VarDecl *InstantiatedFrom = Var->getInstantiatedFromStaticDataMember();
      if (InstantiatedFrom)
        return getVisibilityOf(InstantiatedFrom, kind);
    }

    if (const auto *VTSD = dyn_cast<VarTemplateSpecializationDecl>(Var))
      return getVisibilityOf(VTSD->getSpecializedTemplate()->getTemplatedDecl(),
                             kind);

    return std::nullopt;
  }
  // Also handle function template specializations.
  if (const auto *fn = dyn_cast<FunctionDecl>(ND)) {
    // If the function is a specialization of a template with an
    // explicit visibility attribute, use that.
    if (FunctionTemplateSpecializationInfo *templateInfo
          = fn->getTemplateSpecializationInfo())
      return getVisibilityOf(templateInfo->getTemplate()->getTemplatedDecl(),
                             kind);

    // If the function is a member of a specialization of a class template
    // and the corresponding decl has explicit visibility, use that.
    FunctionDecl *InstantiatedFrom = fn->getInstantiatedFromMemberFunction();
    if (InstantiatedFrom)
      return getVisibilityOf(InstantiatedFrom, kind);

    return std::nullopt;
  }

  // The visibility of a template is stored in the templated decl.
  if (const auto *TD = dyn_cast<TemplateDecl>(ND))
    return getVisibilityOf(TD->getTemplatedDecl(), kind);

  return std::nullopt;
}

std::optional<Visibility>
NamedDecl::getExplicitVisibility(ExplicitVisibilityKind kind) const {
  return getExplicitVisibilityAux(this, kind, false);
}

LinkageInfo LinkageComputer::getLVForClosure(const DeclContext *DC,
                                             Decl *ContextDecl,
                                             LVComputationKind computation) {
  // This lambda has its linkage/visibility determined by its owner.
  const NamedDecl *Owner;
  if (!ContextDecl)
    Owner = dyn_cast<NamedDecl>(DC);
  else if (isa<ParmVarDecl>(ContextDecl))
    Owner =
        dyn_cast<NamedDecl>(ContextDecl->getDeclContext()->getRedeclContext());
  else if (isa<ImplicitConceptSpecializationDecl>(ContextDecl)) {
    // Replace with the concept's owning decl, which is either a namespace or a
    // TU, so this needs a dyn_cast.
    Owner = dyn_cast<NamedDecl>(ContextDecl->getDeclContext());
  } else {
    Owner = cast<NamedDecl>(ContextDecl);
  }

  if (!Owner)
    return LinkageInfo::none();

  // If the owner has a deduced type, we need to skip querying the linkage and
  // visibility of that type, because it might involve this closure type.  The
  // only effect of this is that we might give a lambda VisibleNoLinkage rather
  // than NoLinkage when we don't strictly need to, which is benign.
  auto *VD = dyn_cast<VarDecl>(Owner);
  LinkageInfo OwnerLV =
      VD && VD->getType()->getContainedDeducedType()
          ? computeLVForDecl(Owner, computation, /*IgnoreVarTypeLinkage*/true)
          : getLVForDecl(Owner, computation);

  // A lambda never formally has linkage. But if the owner is externally
  // visible, then the lambda is too. We apply the same rules to blocks.
  if (!isExternallyVisible(OwnerLV.getLinkage()))
    return LinkageInfo::none();
  return LinkageInfo(Linkage::VisibleNone, OwnerLV.getVisibility(),
                     OwnerLV.isVisibilityExplicit());
}

LinkageInfo LinkageComputer::getLVForLocalDecl(const NamedDecl *D,
                                               LVComputationKind computation) {
  if (const auto *Function = dyn_cast<FunctionDecl>(D)) {
    if (Function->isInAnonymousNamespace() &&
        !isFirstInExternCContext(Function))
      return LinkageInfo::internal();

    // This is a "void f();" which got merged with a file static.
    if (Function->getCanonicalDecl()->getStorageClass() == SC_Static)
      return LinkageInfo::internal();

    LinkageInfo LV;
    if (!hasExplicitVisibilityAlready(computation)) {
      if (std::optional<Visibility> Vis =
              getExplicitVisibility(Function, computation))
        LV.mergeVisibility(*Vis, true);
    }

    // Note that Sema::MergeCompatibleFunctionDecls already takes care of
    // merging storage classes and visibility attributes, so we don't have to
    // look at previous decls in here.

    return LV;
  }

  if (const auto *Var = dyn_cast<VarDecl>(D)) {
    if (Var->hasExternalStorage()) {
      if (Var->isInAnonymousNamespace() && !isFirstInExternCContext(Var))
        return LinkageInfo::internal();

      LinkageInfo LV;
      if (Var->getStorageClass() == SC_PrivateExtern)
        LV.mergeVisibility(HiddenVisibility, true);
      else if (!hasExplicitVisibilityAlready(computation)) {
        if (std::optional<Visibility> Vis =
                getExplicitVisibility(Var, computation))
          LV.mergeVisibility(*Vis, true);
      }

      if (const VarDecl *Prev = Var->getPreviousDecl()) {
        LinkageInfo PrevLV = getLVForDecl(Prev, computation);
        if (PrevLV.getLinkage() != Linkage::Invalid)
          LV.setLinkage(PrevLV.getLinkage());
        LV.mergeVisibility(PrevLV);
      }

      return LV;
    }

    if (!Var->isStaticLocal())
      return LinkageInfo::none();
  }

  ASTContext &Context = D->getASTContext();
  if (!Context.getLangOpts().CPlusPlus)
    return LinkageInfo::none();

  const Decl *OuterD = getOutermostFuncOrBlockContext(D);
  if (!OuterD || OuterD->isInvalidDecl())
    return LinkageInfo::none();

  LinkageInfo LV;
  if (const auto *BD = dyn_cast<BlockDecl>(OuterD)) {
    if (!BD->getBlockManglingNumber())
      return LinkageInfo::none();

    LV = getLVForClosure(BD->getDeclContext()->getRedeclContext(),
                         BD->getBlockManglingContextDecl(), computation);
  } else {
    const auto *FD = cast<FunctionDecl>(OuterD);
    if (!FD->isInlined() &&
        !isTemplateInstantiation(FD->getTemplateSpecializationKind()))
      return LinkageInfo::none();

    // If a function is hidden by -fvisibility-inlines-hidden option and
    // is not explicitly attributed as a hidden function,
    // we should not make static local variables in the function hidden.
    LV = getLVForDecl(FD, computation);
    if (isa<VarDecl>(D) && useInlineVisibilityHidden(FD) &&
        !LV.isVisibilityExplicit() &&
        !Context.getLangOpts().VisibilityInlinesHiddenStaticLocalVar) {
      assert(cast<VarDecl>(D)->isStaticLocal());
      // If this was an implicitly hidden inline method, check again for
      // explicit visibility on the parent class, and use that for static locals
      // if present.
      if (const auto *MD = dyn_cast<CXXMethodDecl>(FD))
        LV = getLVForDecl(MD->getParent(), computation);
      if (!LV.isVisibilityExplicit()) {
        Visibility globalVisibility =
            computation.isValueVisibility()
                ? Context.getLangOpts().getValueVisibilityMode()
                : Context.getLangOpts().getTypeVisibilityMode();
        return LinkageInfo(Linkage::VisibleNone, globalVisibility,
                           /*visibilityExplicit=*/false);
      }
    }
  }
  if (!isExternallyVisible(LV.getLinkage()))
    return LinkageInfo::none();
  return LinkageInfo(Linkage::VisibleNone, LV.getVisibility(),
                     LV.isVisibilityExplicit());
}

LinkageInfo LinkageComputer::computeLVForDecl(const NamedDecl *D,
                                              LVComputationKind computation,
                                              bool IgnoreVarTypeLinkage) {
  // Internal_linkage attribute overrides other considerations.
  if (D->hasAttr<InternalLinkageAttr>())
    return LinkageInfo::internal();

  // Objective-C: treat all Objective-C declarations as having external
  // linkage.
  switch (D->getKind()) {
    default:
      break;

    // Per C++ [basic.link]p2, only the names of objects, references,
    // functions, types, templates, namespaces, and values ever have linkage.
    //
    // Note that the name of a typedef, namespace alias, using declaration,
    // and so on are not the name of the corresponding type, namespace, or
    // declaration, so they do *not* have linkage.
    case Decl::ImplicitParam:
    case Decl::Label:
    case Decl::NamespaceAlias:
    case Decl::ParmVar:
    case Decl::Using:
    case Decl::UsingEnum:
    case Decl::UsingShadow:
    case Decl::UsingDirective:
      return LinkageInfo::none();

    case Decl::EnumConstant:
      // C++ [basic.link]p4: an enumerator has the linkage of its enumeration.
      if (D->getASTContext().getLangOpts().CPlusPlus)
        return getLVForDecl(cast<EnumDecl>(D->getDeclContext()), computation);
      return LinkageInfo::visible_none();

    case Decl::Typedef:
    case Decl::TypeAlias:
      // A typedef declaration has linkage if it gives a type a name for
      // linkage purposes.
      if (!cast<TypedefNameDecl>(D)
               ->getAnonDeclWithTypedefName(/*AnyRedecl*/true))
        return LinkageInfo::none();
      break;

    case Decl::TemplateTemplateParm: // count these as external
    case Decl::NonTypeTemplateParm:
    case Decl::ObjCAtDefsField:
    case Decl::ObjCCategory:
    case Decl::ObjCCategoryImpl:
    case Decl::ObjCCompatibleAlias:
    case Decl::ObjCImplementation:
    case Decl::ObjCMethod:
    case Decl::ObjCProperty:
    case Decl::ObjCPropertyImpl:
    case Decl::ObjCProtocol:
      return getExternalLinkageFor(D);

    case Decl::CXXRecord: {
      const auto *Record = cast<CXXRecordDecl>(D);
      if (Record->isLambda()) {
        if (Record->hasKnownLambdaInternalLinkage() ||
            !Record->getLambdaManglingNumber()) {
          // This lambda has no mangling number, so it's internal.
          return LinkageInfo::internal();
        }

        return getLVForClosure(
                  Record->getDeclContext()->getRedeclContext(),
                  Record->getLambdaContextDecl(), computation);
      }

      break;
    }

    case Decl::TemplateParamObject: {
      // The template parameter object can be referenced from anywhere its type
      // and value can be referenced.
      auto *TPO = cast<TemplateParamObjectDecl>(D);
      LinkageInfo LV = getLVForType(*TPO->getType(), computation);
      LV.merge(getLVForValue(TPO->getValue(), computation));
      return LV;
    }
  }

  // Handle linkage for namespace-scope names.
  if (D->getDeclContext()->getRedeclContext()->isFileContext())
    return getLVForNamespaceScopeDecl(D, computation, IgnoreVarTypeLinkage);

  // C++ [basic.link]p5:
  //   In addition, a member function, static data member, a named
  //   class or enumeration of class scope, or an unnamed class or
  //   enumeration defined in a class-scope typedef declaration such
  //   that the class or enumeration has the typedef name for linkage
  //   purposes (7.1.3), has external linkage if the name of the class
  //   has external linkage.
  if (D->getDeclContext()->isRecord())
    return getLVForClassMember(D, computation, IgnoreVarTypeLinkage);

  // C++ [basic.link]p6:
  //   The name of a function declared in block scope and the name of
  //   an object declared by a block scope extern declaration have
  //   linkage. If there is a visible declaration of an entity with
  //   linkage having the same name and type, ignoring entities
  //   declared outside the innermost enclosing namespace scope, the
  //   block scope declaration declares that same entity and receives
  //   the linkage of the previous declaration. If there is more than
  //   one such matching entity, the program is ill-formed. Otherwise,
  //   if no matching entity is found, the block scope entity receives
  //   external linkage.
  if (D->getDeclContext()->isFunctionOrMethod())
    return getLVForLocalDecl(D, computation);

  // C++ [basic.link]p6:
  //   Names not covered by these rules have no linkage.
  return LinkageInfo::none();
}

/// getLVForDecl - Get the linkage and visibility for the given declaration.
LinkageInfo LinkageComputer::getLVForDecl(const NamedDecl *D,
                                          LVComputationKind computation) {
  // Internal_linkage attribute overrides other considerations.
  if (D->hasAttr<InternalLinkageAttr>())
    return LinkageInfo::internal();

  if (computation.IgnoreAllVisibility && D->hasCachedLinkage())
    return LinkageInfo(D->getCachedLinkage(), DefaultVisibility, false);

  if (std::optional<LinkageInfo> LI = lookup(D, computation))
    return *LI;

  LinkageInfo LV = computeLVForDecl(D, computation);
  if (D->hasCachedLinkage())
    assert(D->getCachedLinkage() == LV.getLinkage());

  D->setCachedLinkage(LV.getLinkage());
  cache(D, computation, LV);

#ifndef NDEBUG
  // In C (because of gnu inline) and in c++ with microsoft extensions an
  // static can follow an extern, so we can have two decls with different
  // linkages.
  const LangOptions &Opts = D->getASTContext().getLangOpts();
  if (!Opts.CPlusPlus || Opts.MicrosoftExt)
    return LV;

  // We have just computed the linkage for this decl. By induction we know
  // that all other computed linkages match, check that the one we just
  // computed also does.
  NamedDecl *Old = nullptr;
  for (auto *I : D->redecls()) {
    auto *T = cast<NamedDecl>(I);
    if (T == D)
      continue;
    if (!T->isInvalidDecl() && T->hasCachedLinkage()) {
      Old = T;
      break;
    }
  }
  assert(!Old || Old->getCachedLinkage() == D->getCachedLinkage());
#endif

  return LV;
}

LinkageInfo LinkageComputer::getDeclLinkageAndVisibility(const NamedDecl *D) {
  NamedDecl::ExplicitVisibilityKind EK = usesTypeVisibility(D)
                                             ? NamedDecl::VisibilityForType
                                             : NamedDecl::VisibilityForValue;
  LVComputationKind CK(EK);
  return getLVForDecl(D, D->getASTContext().getLangOpts().IgnoreXCOFFVisibility
                             ? CK.forLinkageOnly()
                             : CK);
}

Module *Decl::getOwningModuleForLinkage(bool IgnoreLinkage) const {
  if (isa<NamespaceDecl>(this))
    // Namespaces never have module linkage.  It is the entities within them
    // that [may] do.
    return nullptr;

  Module *M = getOwningModule();
  if (!M)
    return nullptr;

  switch (M->Kind) {
  case Module::ModuleMapModule:
    // Module map modules have no special linkage semantics.
    return nullptr;

  case Module::ModuleInterfaceUnit:
  case Module::ModuleImplementationUnit:
  case Module::ModulePartitionInterface:
  case Module::ModulePartitionImplementation:
    return M;

  case Module::ModuleHeaderUnit:
  case Module::ExplicitGlobalModuleFragment:
  case Module::ImplicitGlobalModuleFragment: {
    // External linkage declarations in the global module have no owning module
    // for linkage purposes. But internal linkage declarations in the global
    // module fragment of a particular module are owned by that module for
    // linkage purposes.
    // FIXME: p1815 removes the need for this distinction -- there are no
    // internal linkage declarations that need to be referred to from outside
    // this TU.
    if (IgnoreLinkage)
      return nullptr;
    bool InternalLinkage;
    if (auto *ND = dyn_cast<NamedDecl>(this))
      InternalLinkage = !ND->hasExternalFormalLinkage();
    else
      InternalLinkage = isInAnonymousNamespace();
    return InternalLinkage ? M->Kind == Module::ModuleHeaderUnit ? M : M->Parent
                           : nullptr;
  }

  case Module::PrivateModuleFragment:
    // The private module fragment is part of its containing module for linkage
    // purposes.
    return M->Parent;
  }

  llvm_unreachable("unknown module kind");
}

void NamedDecl::printName(raw_ostream &OS, const PrintingPolicy &Policy) const {
  Name.print(OS, Policy);
}

void NamedDecl::printName(raw_ostream &OS) const {
  printName(OS, getASTContext().getPrintingPolicy());
}

std::string NamedDecl::getQualifiedNameAsString() const {
  std::string QualName;
  llvm::raw_string_ostream OS(QualName);
  printQualifiedName(OS, getASTContext().getPrintingPolicy());
  return QualName;
}

void NamedDecl::printQualifiedName(raw_ostream &OS) const {
  printQualifiedName(OS, getASTContext().getPrintingPolicy());
}

void NamedDecl::printQualifiedName(raw_ostream &OS,
                                   const PrintingPolicy &P) const {
  if (getDeclContext()->isFunctionOrMethod()) {
    // We do not print '(anonymous)' for function parameters without name.
    printName(OS, P);
    return;
  }
  printNestedNameSpecifier(OS, P);
  if (getDeclName())
    OS << *this;
  else {
    // Give the printName override a chance to pick a different name before we
    // fall back to "(anonymous)".
    SmallString<64> NameBuffer;
    llvm::raw_svector_ostream NameOS(NameBuffer);
    printName(NameOS, P);
    if (NameBuffer.empty())
      OS << "(anonymous)";
    else
      OS << NameBuffer;
  }
}

void NamedDecl::printNestedNameSpecifier(raw_ostream &OS) const {
  printNestedNameSpecifier(OS, getASTContext().getPrintingPolicy());
}

void NamedDecl::printNestedNameSpecifier(raw_ostream &OS,
                                         const PrintingPolicy &P) const {
  const DeclContext *Ctx = getDeclContext();

  // For ObjC methods and properties, look through categories and use the
  // interface as context.
  if (auto *MD = dyn_cast<ObjCMethodDecl>(this)) {
    if (auto *ID = MD->getClassInterface())
      Ctx = ID;
  } else if (auto *PD = dyn_cast<ObjCPropertyDecl>(this)) {
    if (auto *MD = PD->getGetterMethodDecl())
      if (auto *ID = MD->getClassInterface())
        Ctx = ID;
  } else if (auto *ID = dyn_cast<ObjCIvarDecl>(this)) {
    if (auto *CI = ID->getContainingInterface())
      Ctx = CI;
  }

  if (Ctx->isFunctionOrMethod())
    return;

  using ContextsTy = SmallVector<const DeclContext *, 8>;
  ContextsTy Contexts;

  // Collect named contexts.
  DeclarationName NameInScope = getDeclName();
  for (; Ctx; Ctx = Ctx->getParent()) {
    // Suppress anonymous namespace if requested.
    if (P.SuppressUnwrittenScope && isa<NamespaceDecl>(Ctx) &&
        cast<NamespaceDecl>(Ctx)->isAnonymousNamespace())
      continue;

    // Suppress inline namespace if it doesn't make the result ambiguous.
    if (P.SuppressInlineNamespace && Ctx->isInlineNamespace() && NameInScope &&
        cast<NamespaceDecl>(Ctx)->isRedundantInlineQualifierFor(NameInScope))
      continue;

    // Skip non-named contexts such as linkage specifications and ExportDecls.
    const NamedDecl *ND = dyn_cast<NamedDecl>(Ctx);
    if (!ND)
      continue;

    Contexts.push_back(Ctx);
    NameInScope = ND->getDeclName();
  }

  for (const DeclContext *DC : llvm::reverse(Contexts)) {
    if (const auto *Spec = dyn_cast<ClassTemplateSpecializationDecl>(DC)) {
      OS << Spec->getName();
      const TemplateArgumentList &TemplateArgs = Spec->getTemplateArgs();
      printTemplateArgumentList(
          OS, TemplateArgs.asArray(), P,
          Spec->getSpecializedTemplate()->getTemplateParameters());
    } else if (const auto *ND = dyn_cast<NamespaceDecl>(DC)) {
      if (ND->isAnonymousNamespace()) {
        OS << (P.MSVCFormatting ? "`anonymous namespace\'"
                                : "(anonymous namespace)");
      }
      else
        OS << *ND;
    } else if (const auto *RD = dyn_cast<RecordDecl>(DC)) {
      if (!RD->getIdentifier())
        OS << "(anonymous " << RD->getKindName() << ')';
      else
        OS << *RD;
    } else if (const auto *FD = dyn_cast<FunctionDecl>(DC)) {
      const FunctionProtoType *FT = nullptr;
      if (FD->hasWrittenPrototype())
        FT = dyn_cast<FunctionProtoType>(FD->getType()->castAs<FunctionType>());

      OS << *FD << '(';
      if (FT) {
        unsigned NumParams = FD->getNumParams();
        for (unsigned i = 0; i < NumParams; ++i) {
          if (i)
            OS << ", ";
          OS << FD->getParamDecl(i)->getType().stream(P);
        }

        if (FT->isVariadic()) {
          if (NumParams > 0)
            OS << ", ";
          OS << "...";
        }
      }
      OS << ')';
    } else if (const auto *ED = dyn_cast<EnumDecl>(DC)) {
      // C++ [dcl.enum]p10: Each enum-name and each unscoped
      // enumerator is declared in the scope that immediately contains
      // the enum-specifier. Each scoped enumerator is declared in the
      // scope of the enumeration.
      // For the case of unscoped enumerator, do not include in the qualified
      // name any information about its enum enclosing scope, as its visibility
      // is global.
      if (ED->isScoped())
        OS << *ED;
      else
        continue;
    } else {
      OS << *cast<NamedDecl>(DC);
    }
    OS << "::";
  }
}

void NamedDecl::getNameForDiagnostic(raw_ostream &OS,
                                     const PrintingPolicy &Policy,
                                     bool Qualified) const {
  if (Qualified)
    printQualifiedName(OS, Policy);
  else
    printName(OS, Policy);
}

template<typename T> static bool isRedeclarableImpl(Redeclarable<T> *) {
  return true;
}
static bool isRedeclarableImpl(...) { return false; }
static bool isRedeclarable(Decl::Kind K) {
  switch (K) {
#define DECL(Type, Base) \
  case Decl::Type: \
    return isRedeclarableImpl((Type##Decl *)nullptr);
#define ABSTRACT_DECL(DECL)
#include "clang/AST/DeclNodes.inc"
  }
  llvm_unreachable("unknown decl kind");
}

bool NamedDecl::declarationReplaces(const NamedDecl *OldD,
                                    bool IsKnownNewer) const {
  assert(getDeclName() == OldD->getDeclName() && "Declaration name mismatch");

  // Never replace one imported declaration with another; we need both results
  // when re-exporting.
  if (OldD->isFromASTFile() && isFromASTFile())
    return false;

  // A kind mismatch implies that the declaration is not replaced.
  if (OldD->getKind() != getKind())
    return false;

  // For method declarations, we never replace. (Why?)
  if (isa<ObjCMethodDecl>(this))
    return false;

  // For parameters, pick the newer one. This is either an error or (in
  // Objective-C) permitted as an extension.
  if (isa<ParmVarDecl>(this))
    return true;

  // Inline namespaces can give us two declarations with the same
  // name and kind in the same scope but different contexts; we should
  // keep both declarations in this case.
  if (!this->getDeclContext()->getRedeclContext()->Equals(
          OldD->getDeclContext()->getRedeclContext()))
    return false;

  // Using declarations can be replaced if they import the same name from the
  // same context.
  if (const auto *UD = dyn_cast<UsingDecl>(this)) {
    ASTContext &Context = getASTContext();
    return Context.getCanonicalNestedNameSpecifier(UD->getQualifier()) ==
           Context.getCanonicalNestedNameSpecifier(
               cast<UsingDecl>(OldD)->getQualifier());
  }
  if (const auto *UUVD = dyn_cast<UnresolvedUsingValueDecl>(this)) {
    ASTContext &Context = getASTContext();
    return Context.getCanonicalNestedNameSpecifier(UUVD->getQualifier()) ==
           Context.getCanonicalNestedNameSpecifier(
                        cast<UnresolvedUsingValueDecl>(OldD)->getQualifier());
  }

  if (isRedeclarable(getKind())) {
    if (getCanonicalDecl() != OldD->getCanonicalDecl())
      return false;

    if (IsKnownNewer)
      return true;

    // Check whether this is actually newer than OldD. We want to keep the
    // newer declaration. This loop will usually only iterate once, because
    // OldD is usually the previous declaration.
    for (const auto *D : redecls()) {
      if (D == OldD)
        break;

      // If we reach the canonical declaration, then OldD is not actually older
      // than this one.
      //
      // FIXME: In this case, we should not add this decl to the lookup table.
      if (D->isCanonicalDecl())
        return false;
    }

    // It's a newer declaration of the same kind of declaration in the same
    // scope: we want this decl instead of the existing one.
    return true;
  }

  // In all other cases, we need to keep both declarations in case they have
  // different visibility. Any attempt to use the name will result in an
  // ambiguity if more than one is visible.
  return false;
}

bool NamedDecl::hasLinkage() const {
  switch (getFormalLinkage()) {
  case Linkage::Invalid:
    llvm_unreachable("Linkage hasn't been computed!");
  case Linkage::None:
    return false;
  case Linkage::Internal:
    return true;
  case Linkage::UniqueExternal:
  case Linkage::VisibleNone:
    llvm_unreachable("Non-formal linkage is not allowed here!");
  case Linkage::Module:
  case Linkage::External:
    return true;
  }
  llvm_unreachable("Unhandled Linkage enum");
}

NamedDecl *NamedDecl::getUnderlyingDeclImpl() {
  NamedDecl *ND = this;
  if (auto *UD = dyn_cast<UsingShadowDecl>(ND))
    ND = UD->getTargetDecl();

  if (auto *AD = dyn_cast<ObjCCompatibleAliasDecl>(ND))
    return AD->getClassInterface();

  if (auto *AD = dyn_cast<NamespaceAliasDecl>(ND))
    return AD->getNamespace();

  return ND;
}

bool NamedDecl::isCXXInstanceMember() const {
  if (!isCXXClassMember())
    return false;

  const NamedDecl *D = this;
  if (isa<UsingShadowDecl>(D))
    D = cast<UsingShadowDecl>(D)->getTargetDecl();

  if (isa<FieldDecl>(D) || isa<IndirectFieldDecl>(D) || isa<MSPropertyDecl>(D))
    return true;
  if (const auto *MD = dyn_cast_if_present<CXXMethodDecl>(D->getAsFunction()))
    return MD->isInstance();
  return false;
}

//===----------------------------------------------------------------------===//
// DeclaratorDecl Implementation
//===----------------------------------------------------------------------===//

template <typename DeclT>
static SourceLocation getTemplateOrInnerLocStart(const DeclT *decl) {
  if (decl->getNumTemplateParameterLists() > 0)
    return decl->getTemplateParameterList(0)->getTemplateLoc();
  return decl->getInnerLocStart();
}

SourceLocation DeclaratorDecl::getTypeSpecStartLoc() const {
  TypeSourceInfo *TSI = getTypeSourceInfo();
  if (TSI) return TSI->getTypeLoc().getBeginLoc();
  return SourceLocation();
}

SourceLocation DeclaratorDecl::getTypeSpecEndLoc() const {
  TypeSourceInfo *TSI = getTypeSourceInfo();
  if (TSI) return TSI->getTypeLoc().getEndLoc();
  return SourceLocation();
}

void DeclaratorDecl::setQualifierInfo(NestedNameSpecifierLoc QualifierLoc) {
  if (QualifierLoc) {
    // Make sure the extended decl info is allocated.
    if (!hasExtInfo()) {
      // Save (non-extended) type source info pointer.
      auto *savedTInfo = DeclInfo.get<TypeSourceInfo*>();
      // Allocate external info struct.
      DeclInfo = new (getASTContext()) ExtInfo;
      // Restore savedTInfo into (extended) decl info.
      getExtInfo()->TInfo = savedTInfo;
    }
    // Set qualifier info.
    getExtInfo()->QualifierLoc = QualifierLoc;
  } else if (hasExtInfo()) {
    // Here Qualifier == 0, i.e., we are removing the qualifier (if any).
    getExtInfo()->QualifierLoc = QualifierLoc;
  }
}

void DeclaratorDecl::setTrailingRequiresClause(Expr *TrailingRequiresClause) {
  assert(TrailingRequiresClause);
  // Make sure the extended decl info is allocated.
  if (!hasExtInfo()) {
    // Save (non-extended) type source info pointer.
    auto *savedTInfo = DeclInfo.get<TypeSourceInfo*>();
    // Allocate external info struct.
    DeclInfo = new (getASTContext()) ExtInfo;
    // Restore savedTInfo into (extended) decl info.
    getExtInfo()->TInfo = savedTInfo;
  }
  // Set requires clause info.
  getExtInfo()->TrailingRequiresClause = TrailingRequiresClause;
}

void DeclaratorDecl::setTemplateParameterListsInfo(
    ASTContext &Context, ArrayRef<TemplateParameterList *> TPLists) {
  assert(!TPLists.empty());
  // Make sure the extended decl info is allocated.
  if (!hasExtInfo()) {
    // Save (non-extended) type source info pointer.
    auto *savedTInfo = DeclInfo.get<TypeSourceInfo*>();
    // Allocate external info struct.
    DeclInfo = new (getASTContext()) ExtInfo;
    // Restore savedTInfo into (extended) decl info.
    getExtInfo()->TInfo = savedTInfo;
  }
  // Set the template parameter lists info.
  getExtInfo()->setTemplateParameterListsInfo(Context, TPLists);
}

SourceLocation DeclaratorDecl::getOuterLocStart() const {
  return getTemplateOrInnerLocStart(this);
}

// Helper function: returns true if QT is or contains a type
// having a postfix component.
static bool typeIsPostfix(QualType QT) {
  while (true) {
    const Type* T = QT.getTypePtr();
    switch (T->getTypeClass()) {
    default:
      return false;
    case Type::Pointer:
      QT = cast<PointerType>(T)->getPointeeType();
      break;
    case Type::BlockPointer:
      QT = cast<BlockPointerType>(T)->getPointeeType();
      break;
    case Type::MemberPointer:
      QT = cast<MemberPointerType>(T)->getPointeeType();
      break;
    case Type::LValueReference:
    case Type::RValueReference:
      QT = cast<ReferenceType>(T)->getPointeeType();
      break;
    case Type::PackExpansion:
      QT = cast<PackExpansionType>(T)->getPattern();
      break;
    case Type::Paren:
    case Type::ConstantArray:
    case Type::DependentSizedArray:
    case Type::IncompleteArray:
    case Type::VariableArray:
    case Type::FunctionProto:
    case Type::FunctionNoProto:
      return true;
    }
  }
}

SourceRange DeclaratorDecl::getSourceRange() const {
  SourceLocation RangeEnd = getLocation();
  if (TypeSourceInfo *TInfo = getTypeSourceInfo()) {
    // If the declaration has no name or the type extends past the name take the
    // end location of the type.
    if (!getDeclName() || typeIsPostfix(TInfo->getType()))
      RangeEnd = TInfo->getTypeLoc().getSourceRange().getEnd();
  }
  return SourceRange(getOuterLocStart(), RangeEnd);
}

void QualifierInfo::setTemplateParameterListsInfo(
    ASTContext &Context, ArrayRef<TemplateParameterList *> TPLists) {
  // Free previous template parameters (if any).
  if (NumTemplParamLists > 0) {
    Context.Deallocate(TemplParamLists);
    TemplParamLists = nullptr;
    NumTemplParamLists = 0;
  }
  // Set info on matched template parameter lists (if any).
  if (!TPLists.empty()) {
    TemplParamLists = new (Context) TemplateParameterList *[TPLists.size()];
    NumTemplParamLists = TPLists.size();
    std::copy(TPLists.begin(), TPLists.end(), TemplParamLists);
  }
}

//===----------------------------------------------------------------------===//
// VarDecl Implementation
//===----------------------------------------------------------------------===//

const char *VarDecl::getStorageClassSpecifierString(StorageClass SC) {
  switch (SC) {
  case SC_None:                 break;
  case SC_Auto:                 return "auto";
  case SC_Extern:               return "extern";
  case SC_PrivateExtern:        return "__private_extern__";
  case SC_Register:             return "register";
  case SC_Static:               return "static";
  }

  llvm_unreachable("Invalid storage class");
}

VarDecl::VarDecl(Kind DK, ASTContext &C, DeclContext *DC,
                 SourceLocation StartLoc, SourceLocation IdLoc,
                 const IdentifierInfo *Id, QualType T, TypeSourceInfo *TInfo,
                 StorageClass SC)
    : DeclaratorDecl(DK, DC, IdLoc, Id, T, TInfo, StartLoc),
      redeclarable_base(C) {
  static_assert(sizeof(VarDeclBitfields) <= sizeof(unsigned),
                "VarDeclBitfields too large!");
  static_assert(sizeof(ParmVarDeclBitfields) <= sizeof(unsigned),
                "ParmVarDeclBitfields too large!");
  static_assert(sizeof(NonParmVarDeclBitfields) <= sizeof(unsigned),
                "NonParmVarDeclBitfields too large!");
  AllBits = 0;
  VarDeclBits.SClass = SC;
  // Everything else is implicitly initialized to false.
}

VarDecl *VarDecl::Create(ASTContext &C, DeclContext *DC, SourceLocation StartL,
                         SourceLocation IdL, const IdentifierInfo *Id,
                         QualType T, TypeSourceInfo *TInfo, StorageClass S) {
  return new (C, DC) VarDecl(Var, C, DC, StartL, IdL, Id, T, TInfo, S);
}

VarDecl *VarDecl::CreateDeserialized(ASTContext &C, unsigned ID) {
  return new (C, ID)
      VarDecl(Var, C, nullptr, SourceLocation(), SourceLocation(), nullptr,
              QualType(), nullptr, SC_None);
}

void VarDecl::setStorageClass(StorageClass SC) {
  assert(isLegalForVariable(SC));
  VarDeclBits.SClass = SC;
}

VarDecl::TLSKind VarDecl::getTLSKind() const {
  switch (VarDeclBits.TSCSpec) {
  case TSCS_unspecified:
    if (!hasAttr<ThreadAttr>() &&
        !(getASTContext().getLangOpts().OpenMPUseTLS &&
          getASTContext().getTargetInfo().isTLSSupported() &&
          hasAttr<OMPThreadPrivateDeclAttr>()))
      return TLS_None;
    return ((getASTContext().getLangOpts().isCompatibleWithMSVC(
                LangOptions::MSVC2015)) ||
            hasAttr<OMPThreadPrivateDeclAttr>())
               ? TLS_Dynamic
               : TLS_Static;
  case TSCS___thread: // Fall through.
  case TSCS__Thread_local:
    return TLS_Static;
  case TSCS_thread_local:
    return TLS_Dynamic;
  }
  llvm_unreachable("Unknown thread storage class specifier!");
}

SourceRange VarDecl::getSourceRange() const {
  if (const Expr *Init = getInit()) {
    SourceLocation InitEnd = Init->getEndLoc();
    // If Init is implicit, ignore its source range and fallback on
    // DeclaratorDecl::getSourceRange() to handle postfix elements.
    if (InitEnd.isValid() && InitEnd != getLocation())
      return SourceRange(getOuterLocStart(), InitEnd);
  }
  return DeclaratorDecl::getSourceRange();
}

template<typename T>
static LanguageLinkage getDeclLanguageLinkage(const T &D) {
  // C++ [dcl.link]p1: All function types, function names with external linkage,
  // and variable names with external linkage have a language linkage.
  if (!D.hasExternalFormalLinkage())
    return NoLanguageLinkage;

  // Language linkage is a C++ concept, but saying that everything else in C has
  // C language linkage fits the implementation nicely.
  if (!D.getASTContext().getLangOpts().CPlusPlus)
    return CLanguageLinkage;

  // C++ [dcl.link]p4: A C language linkage is ignored in determining the
  // language linkage of the names of class members and the function type of
  // class member functions.
  const DeclContext *DC = D.getDeclContext();
  if (DC->isRecord())
    return CXXLanguageLinkage;

  // If the first decl is in an extern "C" context, any other redeclaration
  // will have C language linkage. If the first one is not in an extern "C"
  // context, we would have reported an error for any other decl being in one.
  if (isFirstInExternCContext(&D))
    return CLanguageLinkage;
  return CXXLanguageLinkage;
}

template<typename T>
static bool isDeclExternC(const T &D) {
  // Since the context is ignored for class members, they can only have C++
  // language linkage or no language linkage.
  const DeclContext *DC = D.getDeclContext();
  if (DC->isRecord()) {
    assert(D.getASTContext().getLangOpts().CPlusPlus);
    return false;
  }

  return D.getLanguageLinkage() == CLanguageLinkage;
}

LanguageLinkage VarDecl::getLanguageLinkage() const {
  return getDeclLanguageLinkage(*this);
}

bool VarDecl::isExternC() const {
  return isDeclExternC(*this);
}

bool VarDecl::isInExternCContext() const {
  return getLexicalDeclContext()->isExternCContext();
}

bool VarDecl::isInExternCXXContext() const {
  return getLexicalDeclContext()->isExternCXXContext();
}

VarDecl *VarDecl::getCanonicalDecl() { return getFirstDecl(); }

VarDecl::DefinitionKind
VarDecl::isThisDeclarationADefinition(ASTContext &C) const {
  if (isThisDeclarationADemotedDefinition())
    return DeclarationOnly;

  // C++ [basic.def]p2:
  //   A declaration is a definition unless [...] it contains the 'extern'
  //   specifier or a linkage-specification and neither an initializer [...],
  //   it declares a non-inline static data member in a class declaration [...],
  //   it declares a static data member outside a class definition and the variable
  //   was defined within the class with the constexpr specifier [...],
  // C++1y [temp.expl.spec]p15:
  //   An explicit specialization of a static data member or an explicit
  //   specialization of a static data member template is a definition if the
  //   declaration includes an initializer; otherwise, it is a declaration.
  //
  // FIXME: How do you declare (but not define) a partial specialization of
  // a static data member template outside the containing class?
  if (isStaticDataMember()) {
    if (isOutOfLine() &&
        !(getCanonicalDecl()->isInline() &&
          getCanonicalDecl()->isConstexpr()) &&
        (hasInit() ||
         // If the first declaration is out-of-line, this may be an
         // instantiation of an out-of-line partial specialization of a variable
         // template for which we have not yet instantiated the initializer.
         (getFirstDecl()->isOutOfLine()
              ? getTemplateSpecializationKind() == TSK_Undeclared
              : getTemplateSpecializationKind() !=
                    TSK_ExplicitSpecialization) ||
         isa<VarTemplatePartialSpecializationDecl>(this)))
      return Definition;
    if (!isOutOfLine() && isInline())
      return Definition;
    return DeclarationOnly;
  }
  // C99 6.7p5:
  //   A definition of an identifier is a declaration for that identifier that
  //   [...] causes storage to be reserved for that object.
  // Note: that applies for all non-file-scope objects.
  // C99 6.9.2p1:
  //   If the declaration of an identifier for an object has file scope and an
  //   initializer, the declaration is an external definition for the identifier
  if (hasInit())
    return Definition;

  if (hasDefiningAttr())
    return Definition;

  if (const auto *SAA = getAttr<SelectAnyAttr>())
    if (!SAA->isInherited())
      return Definition;

  // A variable template specialization (other than a static data member
  // template or an explicit specialization) is a declaration until we
  // instantiate its initializer.
  if (auto *VTSD = dyn_cast<VarTemplateSpecializationDecl>(this)) {
    if (VTSD->getTemplateSpecializationKind() != TSK_ExplicitSpecialization &&
        !isa<VarTemplatePartialSpecializationDecl>(VTSD) &&
        !VTSD->IsCompleteDefinition)
      return DeclarationOnly;
  }

  if (hasExternalStorage())
    return DeclarationOnly;

  // [dcl.link] p7:
  //   A declaration directly contained in a linkage-specification is treated
  //   as if it contains the extern specifier for the purpose of determining
  //   the linkage of the declared name and whether it is a definition.
  if (isSingleLineLanguageLinkage(*this))
    return DeclarationOnly;

  // C99 6.9.2p2:
  //   A declaration of an object that has file scope without an initializer,
  //   and without a storage class specifier or the scs 'static', constitutes
  //   a tentative definition.
  // No such thing in C++.
  if (!C.getLangOpts().CPlusPlus && isFileVarDecl())
    return TentativeDefinition;

  // What's left is (in C, block-scope) declarations without initializers or
  // external storage. These are definitions.
  return Definition;
}

VarDecl *VarDecl::getActingDefinition() {
  DefinitionKind Kind = isThisDeclarationADefinition();
  if (Kind != TentativeDefinition)
    return nullptr;

  VarDecl *LastTentative = nullptr;

  // Loop through the declaration chain, starting with the most recent.
  for (VarDecl *Decl = getMostRecentDecl(); Decl;
       Decl = Decl->getPreviousDecl()) {
    Kind = Decl->isThisDeclarationADefinition();
    if (Kind == Definition)
      return nullptr;
    // Record the first (most recent) TentativeDefinition that is encountered.
    if (Kind == TentativeDefinition && !LastTentative)
      LastTentative = Decl;
  }

  return LastTentative;
}

VarDecl *VarDecl::getDefinition(ASTContext &C) {
  VarDecl *First = getFirstDecl();
  for (auto *I : First->redecls()) {
    if (I->isThisDeclarationADefinition(C) == Definition)
      return I;
  }
  return nullptr;
}

VarDecl::DefinitionKind VarDecl::hasDefinition(ASTContext &C) const {
  DefinitionKind Kind = DeclarationOnly;

  const VarDecl *First = getFirstDecl();
  for (auto *I : First->redecls()) {
    Kind = std::max(Kind, I->isThisDeclarationADefinition(C));
    if (Kind == Definition)
      break;
  }

  return Kind;
}

const Expr *VarDecl::getAnyInitializer(const VarDecl *&D) const {
  for (auto *I : redecls()) {
    if (auto Expr = I->getInit()) {
      D = I;
      return Expr;
    }
  }
  return nullptr;
}

bool VarDecl::hasInit() const {
  if (auto *P = dyn_cast<ParmVarDecl>(this))
    if (P->hasUnparsedDefaultArg() || P->hasUninstantiatedDefaultArg())
      return false;

  return !Init.isNull();
}

Expr *VarDecl::getInit() {
  if (!hasInit())
    return nullptr;

  if (auto *S = Init.dyn_cast<Stmt *>())
    return cast<Expr>(S);

  auto *Eval = getEvaluatedStmt();
  return cast<Expr>(Eval->Value.isOffset()
                        ? Eval->Value.get(getASTContext().getExternalSource())
                        : Eval->Value.get(nullptr));
}

Stmt **VarDecl::getInitAddress() {
  if (auto *ES = Init.dyn_cast<EvaluatedStmt *>())
    return ES->Value.getAddressOfPointer(getASTContext().getExternalSource());

  return Init.getAddrOfPtr1();
}

VarDecl *VarDecl::getInitializingDeclaration() {
  VarDecl *Def = nullptr;
  for (auto *I : redecls()) {
    if (I->hasInit())
      return I;

    if (I->isThisDeclarationADefinition()) {
      if (isStaticDataMember())
        return I;
      Def = I;
    }
  }
  return Def;
}

bool VarDecl::isOutOfLine() const {
  if (Decl::isOutOfLine())
    return true;

  if (!isStaticDataMember())
    return false;

  // If this static data member was instantiated from a static data member of
  // a class template, check whether that static data member was defined
  // out-of-line.
  if (VarDecl *VD = getInstantiatedFromStaticDataMember())
    return VD->isOutOfLine();

  return false;
}

void VarDecl::setInit(Expr *I) {
  if (auto *Eval = Init.dyn_cast<EvaluatedStmt *>()) {
    Eval->~EvaluatedStmt();
    getASTContext().Deallocate(Eval);
  }

  Init = I;
}

bool VarDecl::mightBeUsableInConstantExpressions(const ASTContext &C) const {
  const LangOptions &Lang = C.getLangOpts();

  // OpenCL permits const integral variables to be used in constant
  // expressions, like in C++98.
  if (!Lang.CPlusPlus && !Lang.OpenCL)
    return false;

  // Function parameters are never usable in constant expressions.
  if (isa<ParmVarDecl>(this))
    return false;

  // The values of weak variables are never usable in constant expressions.
  if (isWeak())
    return false;

  // In C++11, any variable of reference type can be used in a constant
  // expression if it is initialized by a constant expression.
  if (Lang.CPlusPlus11 && getType()->isReferenceType())
    return true;

  // Only const objects can be used in constant expressions in C++. C++98 does
  // not require the variable to be non-volatile, but we consider this to be a
  // defect.
  if (!getType().isConstant(C) || getType().isVolatileQualified())
    return false;

  // In C++, const, non-volatile variables of integral or enumeration types
  // can be used in constant expressions.
  if (getType()->isIntegralOrEnumerationType())
    return true;

  // Additionally, in C++11, non-volatile constexpr variables can be used in
  // constant expressions.
  return Lang.CPlusPlus11 && isConstexpr();
}

bool VarDecl::isUsableInConstantExpressions(const ASTContext &Context) const {
  // C++2a [expr.const]p3:
  //   A variable is usable in constant expressions after its initializing
  //   declaration is encountered...
  const VarDecl *DefVD = nullptr;
  const Expr *Init = getAnyInitializer(DefVD);
  if (!Init || Init->isValueDependent() || getType()->isDependentType())
    return false;
  //   ... if it is a constexpr variable, or it is of reference type or of
  //   const-qualified integral or enumeration type, ...
  if (!DefVD->mightBeUsableInConstantExpressions(Context))
    return false;
  //   ... and its initializer is a constant initializer.
  if (Context.getLangOpts().CPlusPlus && !DefVD->hasConstantInitialization())
    return false;
  // C++98 [expr.const]p1:
  //   An integral constant-expression can involve only [...] const variables
  //   or static data members of integral or enumeration types initialized with
  //   [integer] constant expressions (dcl.init)
  if ((Context.getLangOpts().CPlusPlus || Context.getLangOpts().OpenCL) &&
      !Context.getLangOpts().CPlusPlus11 && !DefVD->hasICEInitializer(Context))
    return false;
  return true;
}

APValue *VarDecl::evaluateValue() const {
  SmallVector<PartialDiagnosticAt, 8> Notes;
  return evaluateValueImpl(Notes, hasConstantInitialization());
}

APValue *VarDecl::evaluateValueImpl(SmallVectorImpl<PartialDiagnosticAt> &Notes,
                                    bool IsConstantInitialization) const {
  EvaluatedStmt *Eval = ensureEvaluatedStmt();

  const auto *Init = getInit();
  assert(!Init->isValueDependent());

  // We only produce notes indicating why an initializer is non-constant the
  // first time it is evaluated. FIXME: The notes won't always be emitted the
  // first time we try evaluation, so might not be produced at all.
  if (Eval->WasEvaluated)
    return Eval->Evaluated.isAbsent() ? nullptr : &Eval->Evaluated;

  if (Eval->IsEvaluating) {
    // FIXME: Produce a diagnostic for self-initialization.
    return nullptr;
  }

  Eval->IsEvaluating = true;

  ASTContext &Ctx = getASTContext();
  bool Result = Init->EvaluateAsInitializer(Eval->Evaluated, Ctx, this, Notes,
                                            IsConstantInitialization);

  // In C++, this isn't a constant initializer if we produced notes. In that
  // case, we can't keep the result, because it may only be correct under the
  // assumption that the initializer is a constant context.
  if (IsConstantInitialization && Ctx.getLangOpts().CPlusPlus &&
      !Notes.empty())
    Result = false;

  // Ensure the computed APValue is cleaned up later if evaluation succeeded,
  // or that it's empty (so that there's nothing to clean up) if evaluation
  // failed.
  if (!Result)
    Eval->Evaluated = APValue();
  else if (Eval->Evaluated.needsCleanup())
    Ctx.addDestruction(&Eval->Evaluated);

  Eval->IsEvaluating = false;
  Eval->WasEvaluated = true;

  return Result ? &Eval->Evaluated : nullptr;
}

APValue *VarDecl::getEvaluatedValue() const {
  if (EvaluatedStmt *Eval = getEvaluatedStmt())
    if (Eval->WasEvaluated)
      return &Eval->Evaluated;

  return nullptr;
}

bool VarDecl::hasICEInitializer(const ASTContext &Context) const {
  const Expr *Init = getInit();
  assert(Init && "no initializer");

  EvaluatedStmt *Eval = ensureEvaluatedStmt();
  if (!Eval->CheckedForICEInit) {
    Eval->CheckedForICEInit = true;
    Eval->HasICEInit = Init->isIntegerConstantExpr(Context);
  }
  return Eval->HasICEInit;
}

bool VarDecl::hasConstantInitialization() const {
  // In C, all globals (and only globals) have constant initialization.
  if (hasGlobalStorage() && !getASTContext().getLangOpts().CPlusPlus)
    return true;

  // In C++, it depends on whether the evaluation at the point of definition
  // was evaluatable as a constant initializer.
  if (EvaluatedStmt *Eval = getEvaluatedStmt())
    return Eval->HasConstantInitialization;

  return false;
}

bool VarDecl::checkForConstantInitialization(
    SmallVectorImpl<PartialDiagnosticAt> &Notes) const {
  EvaluatedStmt *Eval = ensureEvaluatedStmt();
  // If we ask for the value before we know whether we have a constant
  // initializer, we can compute the wrong value (for example, due to
  // std::is_constant_evaluated()).
  assert(!Eval->WasEvaluated &&
         "already evaluated var value before checking for constant init");
  assert(getASTContext().getLangOpts().CPlusPlus && "only meaningful in C++");

  assert(!getInit()->isValueDependent());

  // Evaluate the initializer to check whether it's a constant expression.
  Eval->HasConstantInitialization =
      evaluateValueImpl(Notes, true) && Notes.empty();

  // If evaluation as a constant initializer failed, allow re-evaluation as a
  // non-constant initializer if we later find we want the value.
  if (!Eval->HasConstantInitialization)
    Eval->WasEvaluated = false;

  return Eval->HasConstantInitialization;
}

bool VarDecl::isParameterPack() const {
  return isa<PackExpansionType>(getType());
}

template<typename DeclT>
static DeclT *getDefinitionOrSelf(DeclT *D) {
  assert(D);
  if (auto *Def = D->getDefinition())
    return Def;
  return D;
}

bool VarDecl::isEscapingByref() const {
  return hasAttr<BlocksAttr>() && NonParmVarDeclBits.EscapingByref;
}

bool VarDecl::isNonEscapingByref() const {
  return hasAttr<BlocksAttr>() && !NonParmVarDeclBits.EscapingByref;
}

bool VarDecl::hasDependentAlignment() const {
  QualType T = getType();
  return T->isDependentType() || T->isUndeducedType() ||
         llvm::any_of(specific_attrs<AlignedAttr>(), [](const AlignedAttr *AA) {
           return AA->isAlignmentDependent();
         });
}

VarDecl *VarDecl::getTemplateInstantiationPattern() const {
  const VarDecl *VD = this;

  // If this is an instantiated member, walk back to the template from which
  // it was instantiated.
  if (MemberSpecializationInfo *MSInfo = VD->getMemberSpecializationInfo()) {
    if (isTemplateInstantiation(MSInfo->getTemplateSpecializationKind())) {
      VD = VD->getInstantiatedFromStaticDataMember();
      while (auto *NewVD = VD->getInstantiatedFromStaticDataMember())
        VD = NewVD;
    }
  }

  // If it's an instantiated variable template specialization, find the
  // template or partial specialization from which it was instantiated.
  if (auto *VDTemplSpec = dyn_cast<VarTemplateSpecializationDecl>(VD)) {
    if (isTemplateInstantiation(VDTemplSpec->getTemplateSpecializationKind())) {
      auto From = VDTemplSpec->getInstantiatedFrom();
      if (auto *VTD = From.dyn_cast<VarTemplateDecl *>()) {
        while (!VTD->isMemberSpecialization()) {
          auto *NewVTD = VTD->getInstantiatedFromMemberTemplate();
          if (!NewVTD)
            break;
          VTD = NewVTD;
        }
        return getDefinitionOrSelf(VTD->getTemplatedDecl());
      }
      if (auto *VTPSD =
              From.dyn_cast<VarTemplatePartialSpecializationDecl *>()) {
        while (!VTPSD->isMemberSpecialization()) {
          auto *NewVTPSD = VTPSD->getInstantiatedFromMember();
          if (!NewVTPSD)
            break;
          VTPSD = NewVTPSD;
        }
        return getDefinitionOrSelf<VarDecl>(VTPSD);
      }
    }
  }

  // If this is the pattern of a variable template, find where it was
  // instantiated from. FIXME: Is this necessary?
  if (VarTemplateDecl *VarTemplate = VD->getDescribedVarTemplate()) {
    while (!VarTemplate->isMemberSpecialization()) {
      auto *NewVT = VarTemplate->getInstantiatedFromMemberTemplate();
      if (!NewVT)
        break;
      VarTemplate = NewVT;
    }

    return getDefinitionOrSelf(VarTemplate->getTemplatedDecl());
  }

  if (VD == this)
    return nullptr;
  return getDefinitionOrSelf(const_cast<VarDecl*>(VD));
}
