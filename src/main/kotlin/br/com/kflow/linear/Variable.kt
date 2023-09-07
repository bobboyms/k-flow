package br.com.kflow.linear

class Variable(private val values: List<Number>,
               private val shape: Array<Int>,
               private val name: String = "",
               private var requiresGrad: Boolean = false) : Node<Number>() {

    constructor(value: Number, name: String = "", requiresGrad: Boolean = false) :
            this(listOf(value), arrayOf(1, 1),name,requiresGrad)

    override fun value(): Constant<Number> {
        if (value == null) {
            value = Constant(values=values, shape=shape)
        }
        return value!!
    }

    override fun backward(gradient: Constant<Number>) {
        if (requiresGrad) {
            this.gradient += gradient
        }
    }
}