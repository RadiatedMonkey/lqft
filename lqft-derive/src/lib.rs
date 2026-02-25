use proc_macro::{TokenStream};
use proc_macro2::Span;
use syn::Ident;

#[proc_macro_derive(Observable)]
pub fn derive_observable(stream: TokenStream) -> TokenStream {
    let input = syn::parse_macro_input!(stream as syn::DeriveInput);

    let vis = input.vis;
    let name = input.ident;
    let name_str = name.to_string();
    let state_name = Ident::new(&format!("__{name}ObservableState"), Span::call_site());

    let expanded = quote::quote! {
        impl crate::setup::Observable2 for #name {
            const NAME: &'static str = #name_str;

            
        }
    };

    TokenStream::from(expanded)
}