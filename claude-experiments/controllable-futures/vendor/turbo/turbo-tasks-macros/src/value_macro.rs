use std::sync::OnceLock;

use proc_macro::TokenStream;
use proc_macro2::{Ident, Span};
use quote::{ToTokens, quote, quote_spanned};
use regex::Regex;
use syn::{
    Error, Expr, ExprLit, Fields, FieldsUnnamed, Generics, Item, ItemEnum, ItemStruct, Lit, LitStr,
    Meta, MetaNameValue, Token,
    parse::{Parse, ParseStream},
    parse_macro_input, parse_quote,
    spanned::Spanned,
};

use crate::{global_name::global_name, ident::get_value_type_ident};

enum CellMode {
    Compare,
    New,
}

impl Parse for CellMode {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let ident = input.parse::<LitStr>()?;
        Self::try_from(ident)
    }
}

impl TryFrom<LitStr> for CellMode {
    type Error = Error;

    fn try_from(lit: LitStr) -> Result<Self, Self::Error> {
        match lit.value().as_str() {
            "compare" => Ok(CellMode::Compare),
            "new" => Ok(CellMode::New),
            _ => Err(Error::new_spanned(&lit, "expected \"new\" or \"compare\"")),
        }
    }
}

enum SerializationMode {
    None,
    Auto,
    Custom,
}

impl Parse for SerializationMode {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let ident = input.parse::<LitStr>()?;
        Self::try_from(ident)
    }
}

impl TryFrom<LitStr> for SerializationMode {
    type Error = Error;

    fn try_from(lit: LitStr) -> Result<Self, Self::Error> {
        match lit.value().as_str() {
            "none" => Ok(SerializationMode::None),
            "auto" => Ok(SerializationMode::Auto),
            "custom" => Ok(SerializationMode::Custom),
            _ => Err(Error::new_spanned(
                &lit,
                "expected \"none\", \"auto\", or \"custom\"",
            )),
        }
    }
}

struct ValueArguments {
    serialization_mode: SerializationMode,
    shared: bool,
    cell_mode: CellMode,
    manual_eq: bool,
    transparent: bool,
    /// Should we `#[derive(turbo_tasks::OperationValue)]`?
    operation: Option<Span>,
}

impl Parse for ValueArguments {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let mut result = ValueArguments {
            serialization_mode: SerializationMode::Auto,
            shared: false,
            cell_mode: CellMode::Compare,
            manual_eq: false,
            transparent: false,
            operation: None,
        };
        let punctuated = input.parse_terminated(Meta::parse, Token![,])?;
        for meta in punctuated {
            match (
                meta.path()
                    .get_ident()
                    .map(ToString::to_string)
                    .as_deref()
                    .unwrap_or_default(),
                meta,
            ) {
                ("shared", Meta::Path(_)) => {
                    result.shared = true;
                }
                (
                    "serialization",
                    Meta::NameValue(MetaNameValue {
                        value:
                            Expr::Lit(ExprLit {
                                lit: Lit::Str(str), ..
                            }),
                        ..
                    }),
                ) => {
                    result.serialization_mode = SerializationMode::try_from(str)?;
                }
                (
                    "cell",
                    Meta::NameValue(MetaNameValue {
                        value:
                            Expr::Lit(ExprLit {
                                lit: Lit::Str(str), ..
                            }),
                        ..
                    }),
                ) => {
                    result.cell_mode = CellMode::try_from(str)?;
                }
                (
                    "eq",
                    Meta::NameValue(MetaNameValue {
                        value:
                            Expr::Lit(ExprLit {
                                lit: Lit::Str(str), ..
                            }),
                        ..
                    }),
                ) => {
                    result.manual_eq = if str.value() == "manual" {
                        true
                    } else {
                        return Err(Error::new_spanned(&str, "expected \"manual\""));
                    };
                }
                ("transparent", Meta::Path(_)) => {
                    result.transparent = true;
                }
                ("operation", Meta::Path(path)) => {
                    result.operation = Some(path.span());
                }
                (_, meta) => {
                    return Err(Error::new_spanned(
                        &meta,
                        format!(
                            "unexpected {meta:?}, expected \"shared\", \"into\", \
                             \"serialization\", \"cell\", \"eq\", \"transparent\", or \
                             \"operation\""
                        ),
                    ));
                }
            }
        }

        Ok(result)
    }
}

pub fn value(args: TokenStream, input: TokenStream) -> TokenStream {
    let item = parse_macro_input!(input as Item);
    let ValueArguments {
        serialization_mode,
        shared,
        cell_mode,
        manual_eq,
        transparent,
        operation,
    } = parse_macro_input!(args as ValueArguments);

    let mut struct_attributes = vec![quote! {
        #[derive(
            turbo_tasks::ShrinkToFit,
            turbo_tasks::trace::TraceRawVcs,
            turbo_tasks::NonLocalValue,
        )]
        #[shrink_to_fit(crate = "turbo_tasks::macro_helpers::shrink_to_fit")]
    }];

    let mut inner_type = None;
    if transparent {
        if let Item::Struct(ItemStruct {
            fields: Fields::Unnamed(FieldsUnnamed { unnamed, .. }),
            ..
        }) = &item
            && unnamed.len() == 1
        {
            let field = unnamed.iter().next().unwrap();
            inner_type = Some(field.ty.clone());

            // generate a type string to add to the docs
            let inner_type_string = inner_type.to_token_stream().to_string();

            // HACK: proc_macro2 inserts whitespace between every token. It's ugly, so
            // remove it, assuming these whitespace aren't syntactically important. Using
            // prettyplease (or similar) would be more correct, but slower and add another
            // dependency.
            static WHITESPACE_RE: OnceLock<Regex> = OnceLock::new();
            // Remove whitespace, as long as there is a non-word character (e.g. `>` or `,`)
            // on either side. Try not to remove whitespace between `dyn Trait`.
            let whitespace_re = WHITESPACE_RE
                .get_or_init(|| Regex::new(r"\b \B|\B \b|\B \B").expect("WHITESPACE_RE is valid"));
            let inner_type_string = whitespace_re.replace_all(&inner_type_string, "");

            // Add a couple blank lines in case there's already a doc comment we're
            // effectively appending to. If there's not, rustdoc will strip
            // the leading whitespace.
            let doc_str = format!(
                "\n\nThis is a [transparent value type][turbo_tasks::value#transparent] wrapping \
                 [`{inner_type_string}`].",
            );

            struct_attributes.push(parse_quote! {
                #[doc = #doc_str]
            });
        }
        if inner_type.is_none() {
            item.span()
                .unwrap()
                .error(
                    "#[turbo_tasks::value(transparent)] is only valid with single-item unit \
                     structs",
                )
                .emit();
        }
    }

    let ident = match &item {
        Item::Enum(ItemEnum { ident, .. }) => ident,
        Item::Struct(ItemStruct { ident, .. }) => ident,
        _ => {
            item.span().unwrap().error("unsupported syntax").emit();

            return quote! {
                #item
            }
            .into();
        }
    };

    let (cell_prefix, cell_access_content, read) = if let Some(inner_type) = &inner_type {
        (
            quote! { pub },
            quote! {
                content.0
            },
            quote! {
                turbo_tasks::VcTransparentRead::<#ident, #inner_type, #ident>
            },
        )
    } else {
        (
            if shared {
                quote! { pub }
            } else {
                quote! {}
            },
            quote! { content },
            quote! {
                turbo_tasks::VcDefaultRead::<#ident>
            },
        )
    };

    let cell_mode = match cell_mode {
        CellMode::New => quote! {
            turbo_tasks::VcCellNewMode<#ident>
        },
        CellMode::Compare => quote! {
            turbo_tasks::VcCellCompareMode<#ident>
        },
    };

    let cell_struct = quote! {
        /// Places a value in a cell of the current task.
        ///
        /// Cell is selected based on the value type and call order of `cell`.
        #cell_prefix fn cell(self) -> turbo_tasks::Vc<Self> {
            let content = self;
            turbo_tasks::Vc::cell_private(#cell_access_content)
        }

        /// Places a value in a cell of the current task. Returns a
        /// [`ResolvedVc`][turbo_tasks::ResolvedVc].
        ///
        /// Cell is selected based on the value type and call order of `cell`.
        #cell_prefix fn resolved_cell(self) -> turbo_tasks::ResolvedVc<Self> {
            let content = self;
            turbo_tasks::ResolvedVc::cell_private(#cell_access_content)
        }
    };

    match serialization_mode {
        SerializationMode::Auto => {
            struct_attributes.push(quote! {
                #[derive(
                    turbo_tasks::macro_helpers::bincode::Encode,
                    turbo_tasks::macro_helpers::bincode::Decode,
                )]
                #[bincode(crate = "turbo_tasks::macro_helpers::bincode")]
            });
        }
        SerializationMode::None | SerializationMode::Custom => {}
    };
    if inner_type.is_some() {
        // Transparent structs have their own manual `ValueDebug` implementation.
        struct_attributes.push(quote! {
            #[repr(transparent)]
        });
    } else {
        struct_attributes.push(quote! {
            #[derive(
                turbo_tasks::debug::ValueDebugFormat,
                turbo_tasks::debug::internal::ValueDebug,
            )]
        });
    }
    if !manual_eq {
        struct_attributes.push(quote! {
            #[derive(PartialEq, Eq)]
        });
    }
    if let Some(span) = operation {
        struct_attributes.push(quote_spanned! {
            span =>
            #[derive(turbo_tasks::OperationValue)]
        });
    }

    let name = global_name(quote! {stringify!(#ident) });
    let debug_any_closure = generate_debug_any_closure(ident, &item);
    let new_value_type = match serialization_mode {
        SerializationMode::None => quote! {
            turbo_tasks::ValueType::new::<#ident>(#name).with_debug_any(
                #debug_any_closure
            )
        },
        SerializationMode::Auto | SerializationMode::Custom => {
            quote! {
                turbo_tasks::ValueType::new_with_bincode::<#ident>(#name).with_debug_any(
                    #debug_any_closure
                )
            }
        }
    };
    let has_serialization = match serialization_mode {
        SerializationMode::None => quote! { false },
        SerializationMode::Auto | SerializationMode::Custom => quote! { true },
    };

    let value_debug_impl = if inner_type.is_some() {
        // For transparent values, we defer directly to the inner type's `ValueDebug`
        // implementation.
        quote! {
            #[turbo_tasks::value_impl]
            impl turbo_tasks::debug::ValueDebug for #ident {
                #[turbo_tasks::function]
                async fn dbg(&self) -> anyhow::Result<turbo_tasks::Vc<turbo_tasks::debug::ValueDebugString>> {
                    use turbo_tasks::debug::ValueDebugFormat;
                    (&self.0).value_debug_format(usize::MAX).try_to_value_debug_string().await
                }

                #[turbo_tasks::function]
                async fn dbg_depth(&self, depth: usize) -> anyhow::Result<turbo_tasks::Vc<turbo_tasks::debug::ValueDebugString>> {
                    use turbo_tasks::debug::ValueDebugFormat;
                    (&self.0).value_debug_format(depth).try_to_value_debug_string().await
                }
            }
        }
    } else {
        quote! {}
    };

    let value_type_and_register_code = value_type_and_register(
        ident,
        quote! { #ident },
        None,
        read,
        cell_mode,
        new_value_type,
        has_serialization,
    );

    let expanded = quote! {
        #(#struct_attributes)*
        #item

        impl #ident {
            #cell_struct
        }

        #value_type_and_register_code

        #value_debug_impl
    };

    expanded.into()
}

/// Check if any field in a struct or enum variant has `#[cfg(...)]` attributes.
/// When fields are conditionally compiled, we can't safely generate per-field
/// debug code because field indices and types may change at compile time.
fn has_cfg_fields(item: &Item) -> bool {
    let check_fields = |fields: &Fields| -> bool {
        match fields {
            Fields::Named(named) => named.named.iter().any(|f| {
                f.attrs.iter().any(|a| a.path().is_ident("cfg"))
            }),
            Fields::Unnamed(unnamed) => unnamed.unnamed.iter().any(|f| {
                f.attrs.iter().any(|a| a.path().is_ident("cfg"))
            }),
            Fields::Unit => false,
        }
    };

    match item {
        Item::Struct(s) => check_fields(&s.fields),
        Item::Enum(e) => e.variants.iter().any(|v| {
            // Check for #[cfg] on the variant itself or on its fields
            v.attrs.iter().any(|a| a.path().is_ident("cfg")) || check_fields(&v.fields)
        }),
        _ => false,
    }
}

/// Generate a per-field debug closure for `ValueType.debug_any`.
///
/// For structs with named fields, generates a closure that downcasts `&dyn Any`
/// to the concrete type and formats each field individually. Each field uses
/// autoref specialization: `DebugAnySpecialize<FieldType>` resolves to the
/// inherent `format_value` (returning `Debug` output) when `FieldType: Debug`,
/// or falls through to `DebugAnyFallback::format_value` (returning `<TypeName>`)
/// otherwise. This means we get meaningful output even when the struct itself
/// doesn't implement `Debug`, as long as individual fields do.
fn generate_debug_any_closure(ident: &Ident, item: &Item) -> proc_macro2::TokenStream {
    // Check if any field has #[cfg] attributes — if so, field indices/types can
    // change at compile time and we can't safely generate per-field code.
    if has_cfg_fields(item) {
        return quote! {
            turbo_tasks::macro_helpers::DebugAnySpecialize::<#ident>(std::marker::PhantomData).debug_any_fn()
        };
    }

    match item {
        Item::Struct(ItemStruct { fields, .. }) => match fields {
            Fields::Named(named) => {
                let field_formatters: Vec<proc_macro2::TokenStream> = named
                    .named
                    .iter()
                    .map(|f| {
                        let field_name = f.ident.as_ref().unwrap();
                        let field_name_str = field_name.to_string();
                        let field_ty = &f.ty;
                        quote! {
                            __s.push_str(&format!(
                                "    {}: {},\n",
                                #field_name_str,
                                turbo_tasks::macro_helpers::DebugAnySpecialize::<#field_ty>(
                                    std::marker::PhantomData
                                ).format_value(&__v.#field_name)
                            ));
                        }
                    })
                    .collect();

                let ident_str = ident.to_string();
                quote! {
                    Some(|__any: &dyn std::any::Any| -> String {
                        match __any.downcast_ref::<#ident>() {
                            Some(__v) => {
                                let mut __s = String::from(concat!(#ident_str, " {\n"));
                                #(#field_formatters)*
                                __s.push('}');
                                __s
                            }
                            None => "<downcast failed>".to_string(),
                        }
                    })
                }
            }
            Fields::Unnamed(unnamed) => {
                let field_formatters: Vec<proc_macro2::TokenStream> = unnamed
                    .unnamed
                    .iter()
                    .enumerate()
                    .map(|(i, f)| {
                        let idx = syn::Index::from(i);
                        let field_ty = &f.ty;
                        quote! {
                            if #i > 0 { __s.push_str(", "); }
                            __s.push_str(&turbo_tasks::macro_helpers::DebugAnySpecialize::<#field_ty>(
                                std::marker::PhantomData
                            ).format_value(&__v.#idx));
                        }
                    })
                    .collect();

                let ident_str = ident.to_string();
                quote! {
                    Some(|__any: &dyn std::any::Any| -> String {
                        match __any.downcast_ref::<#ident>() {
                            Some(__v) => {
                                let mut __s = String::from(concat!(#ident_str, "("));
                                #(#field_formatters)*
                                __s.push(')');
                                __s
                            }
                            None => "<downcast failed>".to_string(),
                        }
                    })
                }
            }
            Fields::Unit => {
                let ident_str = ident.to_string();
                quote! {
                    Some(|__any: &dyn std::any::Any| -> String {
                        match __any.downcast_ref::<#ident>() {
                            Some(_) => #ident_str.to_string(),
                            None => "<downcast failed>".to_string(),
                        }
                    })
                }
            }
        },
        Item::Enum(ItemEnum { variants, .. }) => {
            let variant_arms: Vec<proc_macro2::TokenStream> = variants
                .iter()
                .map(|v| {
                    let variant_ident = &v.ident;
                    let variant_str = variant_ident.to_string();
                    match &v.fields {
                        Fields::Named(named) => {
                            let field_names: Vec<&Ident> = named
                                .named
                                .iter()
                                .map(|f| f.ident.as_ref().unwrap())
                                .collect();
                            let field_formatters: Vec<proc_macro2::TokenStream> = named
                                .named
                                .iter()
                                .map(|f| {
                                    let field_name = f.ident.as_ref().unwrap();
                                    let field_name_str = field_name.to_string();
                                    let field_ty = &f.ty;
                                    quote! {
                                        __s.push_str(&format!(
                                            "    {}: {},\n",
                                            #field_name_str,
                                            turbo_tasks::macro_helpers::DebugAnySpecialize::<#field_ty>(
                                                std::marker::PhantomData
                                            ).format_value(#field_name)
                                        ));
                                    }
                                })
                                .collect();

                            quote! {
                                #ident::#variant_ident { #(#field_names),* } => {
                                    let mut __s = String::from(concat!(#variant_str, " {\n"));
                                    #(#field_formatters)*
                                    __s.push('}');
                                    __s
                                }
                            }
                        }
                        Fields::Unnamed(unnamed) => {
                            let binding_names: Vec<Ident> = (0..unnamed.unnamed.len())
                                .map(|i| Ident::new(&format!("__f{i}"), Span::call_site()))
                                .collect();
                            let field_formatters: Vec<proc_macro2::TokenStream> = unnamed
                                .unnamed
                                .iter()
                                .enumerate()
                                .map(|(i, f)| {
                                    let binding = &binding_names[i];
                                    let field_ty = &f.ty;
                                    quote! {
                                        if #i > 0 { __s.push_str(", "); }
                                        __s.push_str(&turbo_tasks::macro_helpers::DebugAnySpecialize::<#field_ty>(
                                            std::marker::PhantomData
                                        ).format_value(#binding));
                                    }
                                })
                                .collect();

                            quote! {
                                #ident::#variant_ident(#(#binding_names),*) => {
                                    let mut __s = String::from(concat!(#variant_str, "("));
                                    #(#field_formatters)*
                                    __s.push(')');
                                    __s
                                }
                            }
                        }
                        Fields::Unit => {
                            quote! {
                                #ident::#variant_ident => #variant_str.to_string(),
                            }
                        }
                    }
                })
                .collect();

            quote! {
                Some(|__any: &dyn std::any::Any| -> String {
                    match __any.downcast_ref::<#ident>() {
                        Some(__v) => match __v {
                            #(#variant_arms)*
                        },
                        None => "<downcast failed>".to_string(),
                    }
                })
            }
        }
        _ => {
            // Fallback — try whole-type Debug via autoref specialization
            quote! {
                turbo_tasks::macro_helpers::DebugAnySpecialize::<#ident>(std::marker::PhantomData).debug_any_fn()
            }
        }
    }
}

pub fn value_type_and_register(
    ident: &Ident,
    ty: proc_macro2::TokenStream,
    generics: Option<&Generics>,
    read: proc_macro2::TokenStream,
    cell_mode: proc_macro2::TokenStream,
    new_value_type: proc_macro2::TokenStream,
    has_serialization: proc_macro2::TokenStream,
) -> proc_macro2::TokenStream {
    let value_type_ident = get_value_type_ident(ident);

    let (impl_generics, where_clause) = if let Some(generics) = generics {
        let (impl_generics, _, where_clause) = generics.split_for_impl();
        (quote! { #impl_generics }, quote! { #where_clause })
    } else {
        (quote!(), quote!())
    };

    quote! {

        static #value_type_ident: turbo_tasks::macro_helpers::Lazy<turbo_tasks::ValueType> =
            turbo_tasks::macro_helpers::Lazy::new(|| {
                let mut value = #new_value_type;
                turbo_tasks::macro_helpers::register_trait_methods(&mut value);
                value
             });

        turbo_tasks::macro_helpers::inventory_submit!{turbo_tasks::macro_helpers::CollectableValueType(&#value_type_ident)}

        unsafe impl #impl_generics turbo_tasks::VcValueType for #ty #where_clause {
            type Read = #read;
            type CellMode = #cell_mode;

            fn get_value_type_id() -> turbo_tasks::ValueTypeId {
                static ident: turbo_tasks::macro_helpers::Lazy<turbo_tasks::ValueTypeId> =
                    turbo_tasks::macro_helpers::Lazy::new(|| {
                        turbo_tasks::registry::get_value_type_id(&#value_type_ident)
                    });

                *ident
            }

            fn has_serialization() -> bool {
                #has_serialization
            }
        }
    }
}
