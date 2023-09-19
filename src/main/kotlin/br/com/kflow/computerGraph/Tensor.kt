package br.com.kflow.computerGraph

import br.com.kflow.value.Value

class Tensor(private val values: List<Number>,
             private val shape: Array<Int>,
             private val name: String = "",
             private val constant: Boolean = false,
             private var requiresGrad: Boolean = false) : Node<Number>() {

    constructor(value: Number, name: String = "", constant: Boolean = false, requiresGrad: Boolean = false) :
            this(listOf(value), arrayOf(1, 1),name,constant,requiresGrad)

    init {
        if (constant) {
            defineAsConstant()
        }
    }

    override fun value(): Value<Number> {
        if (value == null) {
            value = Value(values=values, shape=shape)
        }
        return value!!
    }

    override fun backward(gradient: Value<Number>) {
        println("ssss")
        gradient.printMatrix()
        if (requiresGrad) {
            println("caiu aqui")
            this.gradient += gradient
        }
    }
}