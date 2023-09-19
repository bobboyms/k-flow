package br.com.kflow.activations

import br.com.kflow.computerGraph.*

fun softmax(variable:Node<Number>):Node<Number> {
    val e = Exp(variable)
    val r = e / Sum(e)
    r.defineAsConstant()
    return r
}