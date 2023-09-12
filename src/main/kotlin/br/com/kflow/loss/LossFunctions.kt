package br.com.kflow.loss

import br.com.kflow.linear.*

fun quadratic(actual:Node<Number>, predicted:Node<Number>):Node<Number> {
    return Pow((actual - predicted), Constant(2))
}

fun MSE(actual:Node<Number>, predicted:Node<Number>):Node<Number> {
    return Sum(quadratic(actual,predicted)) / Variable(actual.value().size().toFloat())
}