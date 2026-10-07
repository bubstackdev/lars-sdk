//! `Lars::ask`: several predicates in one request, so the model reads the input once.

use std::sync::Arc;

use serde_json::Value;

use crate::predicate::{Kind, Predicate};
use crate::{Content, Error, Result};

/// Predicates `Lars::ask` can put to the model together: a tuple of predicate references (of any
/// kinds; up to 8), or an array, slice or `Vec` of predicates of one kind. The decisions come back
/// in the same shape.
pub trait Questions<I: ?Sized> {
    type Decisions: Send + 'static;

    #[doc(hidden)]
    fn parts(&self) -> Vec<Part<'_, I>>;
    #[doc(hidden)]
    fn decoder(&self) -> Decoder<Self::Decisions>;
}

#[doc(hidden)]
pub struct Part<'a, I: ?Sized> {
    pub(crate) name: &'a str,
    pub(crate) question: &'a Value,
    pub(crate) state: &'a (dyn Fn(&I) -> Content + Send + Sync),
    pub(crate) state_version: Option<&'a str>,
}

/// Reads the decisions out of a response's `answers`, given `accepts(name, confidence)`.
#[doc(hidden)]
pub type Decoder<T> = Box<dyn FnOnce(&Value, &dyn Fn(&str, f64) -> bool) -> Result<T> + Send>;

fn part<K: Kind, I: ?Sized>(p: &Predicate<K, I>) -> Part<'_, I> {
    Part { name: &p.name, question: &p.question, state: &*p.state, state_version: p.state_version.as_deref() }
}

fn decode<K: Kind>(
    (name, kind): &(Arc<str>, Arc<K>),
    answers: &Value,
    accepts: &dyn Fn(&str, f64) -> bool,
) -> Result<K::Decision> {
    let answer = answers.get(&**name).ok_or_else(|| Error::internal(format!("no answer for {name}")))?;
    kind.decision(answer, &|confidence| accepts(name, confidence))
}

fn decode_all<K: Kind>(parts: Vec<(Arc<str>, Arc<K>)>) -> Decoder<Vec<K::Decision>> {
    Box::new(move |answers, accepts| parts.iter().map(|p| decode(p, answers, accepts)).collect())
}

macro_rules! tuple {
    ($($K:ident $i:tt),+) => {
        impl<'a, I: ?Sized, $($K: Kind),+> Questions<I> for ($(&'a Predicate<$K, I>,)+) {
            type Decisions = ($($K::Decision,)+);

            fn parts(&self) -> Vec<Part<'_, I>> {
                vec![$(part(self.$i)),+]
            }

            fn decoder(&self) -> Decoder<Self::Decisions> {
                let parts = ($((self.$i.name.clone(), self.$i.kind.clone()),)+);
                Box::new(move |answers, accepts| Ok(($(decode(&parts.$i, answers, accepts)?,)+)))
            }
        }
    };
}

tuple!(A 0);
tuple!(A 0, B 1);
tuple!(A 0, B 1, C 2);
tuple!(A 0, B 1, C 2, D 3);
tuple!(A 0, B 1, C 2, D 3, E 4);
tuple!(A 0, B 1, C 2, D 3, E 4, F 5);
tuple!(A 0, B 1, C 2, D 3, E 4, F 5, G 6);
tuple!(A 0, B 1, C 2, D 3, E 4, F 5, G 6, H 7);

impl<I: ?Sized, K: Kind, const N: usize> Questions<I> for [&Predicate<K, I>; N] {
    type Decisions = Vec<K::Decision>;

    fn parts(&self) -> Vec<Part<'_, I>> {
        self.iter().map(|p| part(p)).collect()
    }

    fn decoder(&self) -> Decoder<Self::Decisions> {
        decode_all(self.iter().map(|p| (p.name.clone(), p.kind.clone())).collect())
    }
}

impl<I: ?Sized, K: Kind> Questions<I> for &[Predicate<K, I>] {
    type Decisions = Vec<K::Decision>;

    fn parts(&self) -> Vec<Part<'_, I>> {
        self.iter().map(part).collect()
    }

    fn decoder(&self) -> Decoder<Self::Decisions> {
        decode_all(self.iter().map(|p| (p.name.clone(), p.kind.clone())).collect())
    }
}

impl<I: ?Sized, K: Kind> Questions<I> for &Vec<Predicate<K, I>> {
    type Decisions = Vec<K::Decision>;

    fn parts(&self) -> Vec<Part<'_, I>> {
        self.iter().map(part).collect()
    }

    fn decoder(&self) -> Decoder<Self::Decisions> {
        decode_all(self.iter().map(|p| (p.name.clone(), p.kind.clone())).collect())
    }
}
