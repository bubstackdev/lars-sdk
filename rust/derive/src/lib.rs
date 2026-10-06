//! `#[derive(Options)]` for moedex-lars: a fieldless enum's variants become the options of a
//! choice or multi. Use it through `moedex_lars::Options`.
//!
//! Each variant's key is its name in snake_case, or `#[option(key = "...")]`. Its description,
//! the text the model reads, is its doc comment, or `#[option("...")]`.

use proc_macro::TokenStream;
use proc_macro2::TokenStream as Tokens;
use quote::quote;
use syn::parse::ParseStream;
use syn::{Data, DeriveInput, Expr, ExprLit, Fields, Ident, Lit, LitStr, Meta, Token, parse_macro_input};

#[proc_macro_derive(Options, attributes(option))]
pub fn derive_options(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    expand(&input).unwrap_or_else(syn::Error::into_compile_error).into()
}

fn expand(input: &DeriveInput) -> syn::Result<Tokens> {
    let Data::Enum(data) = &input.data else {
        return Err(syn::Error::new_spanned(&input.ident, "Options derives only on enums"));
    };
    if !input.generics.params.is_empty() {
        return Err(syn::Error::new_spanned(&input.generics, "Options can't derive on a generic enum"));
    }
    let (mut idents, mut keys, mut descriptions) = (vec![], vec![], vec![]);
    for variant in &data.variants {
        if !matches!(variant.fields, Fields::Unit) {
            return Err(syn::Error::new_spanned(variant, "an option can't carry data"));
        }
        let (mut key, mut description, mut docs) = (None, None, vec![]);
        for attr in &variant.attrs {
            if attr.path().is_ident("doc") {
                if let Meta::NameValue(nv) = &attr.meta {
                    let Expr::Lit(ExprLit { lit: Lit::Str(text), .. }) = &nv.value else { continue };
                    let line = text.value().trim().to_string();
                    if !line.is_empty() {
                        docs.push(line);
                    }
                }
            } else if attr.path().is_ident("option") {
                attr.parse_args_with(|input: ParseStream| {
                    while !input.is_empty() {
                        if input.peek(LitStr) {
                            description = Some(input.parse::<LitStr>()?.value());
                        } else {
                            let name: Ident = input.parse()?;
                            if name != "key" {
                                return Err(syn::Error::new_spanned(name, "expected a description or key = \"...\""));
                            }
                            input.parse::<Token![=]>()?;
                            key = Some(input.parse::<LitStr>()?.value());
                        }
                        if !input.is_empty() {
                            input.parse::<Token![,]>()?;
                        }
                    }
                    Ok(())
                })?;
            }
        }
        let key = key.unwrap_or_else(|| snake_case(&variant.ident.to_string()));
        if keys.contains(&key) {
            return Err(syn::Error::new_spanned(variant, format!("two options have the key {key:?}")));
        }
        let description = description.or_else(|| (!docs.is_empty()).then(|| docs.join(" ")));
        idents.push(&variant.ident);
        keys.push(key);
        descriptions.push(match description {
            Some(text) => quote!(::std::option::Option::Some(#text)),
            None => quote!(::std::option::Option::None),
        });
    }
    let name = &input.ident;
    Ok(quote! {
        impl ::moedex_lars::Key for #name {
            fn key(&self) -> ::std::string::String {
                ::std::string::String::from(match self { #(Self::#idents => #keys,)* })
            }

            fn from_key(key: &str) -> ::std::option::Option<Self> {
                match key {
                    #(#keys => ::std::option::Option::Some(Self::#idents),)*
                    _ => ::std::option::Option::None,
                }
            }
        }

        impl ::moedex_lars::Options for #name {
            fn options() -> ::std::vec::Vec<(&'static str, ::std::option::Option<&'static str>)> {
                ::std::vec![#((#keys, #descriptions),)*]
            }
        }
    })
}

/// `FrontDesk` → `front_desk`, as the C# SDK keys enum members.
fn snake_case(name: &str) -> String {
    let mut out = String::new();
    for (i, c) in name.chars().enumerate() {
        if c.is_uppercase() && i > 0 {
            out.push('_');
        }
        out.extend(c.to_lowercase());
    }
    out
}
