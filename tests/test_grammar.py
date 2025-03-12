from typing import Annotated, Union, Literal
from pydantic import BaseModel, Field
from xverify.xml import generate_gbnf_grammar_and_documentation


class A(BaseModel):
    type: Literal["a"]
    a: int


class B(BaseModel):
    type: Literal["b"]
    b: int


class Foo(BaseModel):
    a: A


class MultiFoo(BaseModel):
    a: list[Foo]


class MultiUnionFoo(BaseModel):
    out: list[Annotated[Union[A, B], Field(discriminator="type")]]


def test_simple_model():
    grammar, doc = generate_gbnf_grammar_and_documentation([Foo])
    assert grammar is not None
    assert doc is not None
    assert isinstance(grammar, str)
    assert isinstance(doc, str)


def test_model_with_list():
    grammar, doc = generate_gbnf_grammar_and_documentation([MultiFoo])
    assert grammar is not None
    assert doc is not None
    assert isinstance(grammar, str)
    assert isinstance(doc, str)


def test_model_with_union_list():
    grammar, doc = generate_gbnf_grammar_and_documentation([MultiUnionFoo])
    assert grammar is not None
    assert doc is not None
    assert isinstance(grammar, str)
    assert isinstance(doc, str)

