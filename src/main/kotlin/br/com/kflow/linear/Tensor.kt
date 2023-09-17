package br.com.kflow.linear

import br.com.kflow.value.Value

class Tensor(private val values: List<Number>,
             private val shape: Array<Int>,
             private val name: String = "",
             private var requiresGrad: Boolean = false) : Node<Number>() {

    constructor(value: Number, name: String = "", requiresGrad: Boolean = false) :
            this(listOf(value), arrayOf(1, 1),name,requiresGrad)

    override fun value(): Value<Number> {
        if (value == null) {
            value = Value(values=values, shape=shape)
        }
        return value!!
    }

    override fun backward(gradient: Value<Number>) {
        if (requiresGrad) {
            this.gradient += gradient
        }
    }
}