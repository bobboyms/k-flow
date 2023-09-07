package br.com.kflow.linear

// Use Node<Number> to specify that these operations apply to Nodes with Number types
class Add(private val nodeA: Node<Number>, private val nodeB: Node<Number>) : Node<Number>() {

    override fun value(): Constant<Number> {
        if (value == null) {
            value = nodeA.value() + nodeB.value()
        }
        return value!!
    }

    override fun zeroGrad() {
        super.zeroGrad()
        nodeA.zeroGrad()
        nodeB.zeroGrad()
    }

    override fun backward(gradient: Constant<Number>) {
        this.gradient += gradient
        nodeA.backward(gradient)  // Adjust as necessary
        nodeB.backward(gradient)  // Adjust as necessary
    }
}

class Exp(private val nodeA: Node<Number>) : Node<Number>() {

    override fun value(): Constant<Number> {
        if (value == null) {
            value = Constant(values = nodeA.value().exp().values(), shape = nodeA.value().shape())
        }
        return value!!
    }

    override fun zeroGrad() {
        super.zeroGrad()
        nodeA.zeroGrad()
    }

    override fun backward(gradient: Constant<Number>) {
        this.gradient += gradient
        nodeA.backward(value!! * gradient)
    }

}

class Sum(private val nodeA: Node<Number>) : Node<Number>() {

    override fun value(): Constant<Number> {
        if (value == null) {

            val tempValue = nodeA.value().values().reduce { acc, number ->  acc.toDouble() + number.toDouble()}
                .toT(nodeA.value().values()[0]::class.java)
            value = Constant(tempValue)
        }
        return value!!
    }

    override fun zeroGrad() {
        super.zeroGrad()
        nodeA.zeroGrad()
    }

    override fun backward(gradient: Constant<Number>) {
        this.gradient += gradient
        val consA = Constant(values = List(nodeA.value().size()) { 1.0 as Number }, shape = nodeA.value().shape())

        nodeA.backward(consA * gradient)
    }

}

class Sub(private val nodeA: Node<Number>, private val nodeB: Node<Number>) : Node<Number>() {

    override fun value(): Constant<Number> {
        if (value == null) {
            value = nodeA.value() - nodeB.value()
        }
        return value!!
    }

    override fun zeroGrad() {
        super.zeroGrad()
        nodeA.zeroGrad()
        nodeB.zeroGrad()
    }

    override fun backward(gradient: Constant<Number>) {
        this.gradient += gradient

        val consA = Constant(values = List(nodeA.value().size()) { 1.0 as Number }, shape = nodeA.value().shape())
        val consB = Constant(values = List(nodeB.value().size()) { -1.0 as Number }, shape = nodeB.value().shape())

        nodeA.backward(consA * gradient)
        nodeB.backward(consB * gradient)
    }
}

class Multiply(private val nodeA: Node<Number>, private val nodeB: Node<Number>) : Node<Number>() {

    override fun value(): Constant<Number> {
        if (value == null) {
            value = nodeA.value() * nodeB.value()
        }
        return value!!
    }

    override fun zeroGrad() {
        super.zeroGrad()
        nodeA.zeroGrad()
        nodeB.zeroGrad()
    }

    override fun backward(gradient: Constant<Number>) {
        this.gradient += gradient
        nodeA.backward(nodeB.value() * gradient)  // Derivada parcial em relação a 'a' é 'b'
        nodeB.backward(nodeA.value() * gradient)  // Derivada parcial em relação a 'b' é 'a'
    }
}

class Pow(private val nodeA: Node<Number>, private val power: Constant<Number>) : Node<Number>() {
    override fun value(): Constant<Number> {
        if (value == null) {
            value = nodeA.value().pow(power)
        }
        return value!!
    }

    override fun zeroGrad() {
        super.zeroGrad()
        nodeA.zeroGrad()
    }

    override fun backward(gradient: Constant<Number>) {
        this.gradient += gradient

        val diff = power * nodeA.value().pow(power - Constant(1))
        nodeA.backward(diff * gradient)
    }
}

class Matmul(private val nodeA: Node<Number>, private val nodeB: Node<Number>) : Node<Number>() {

    override fun value(): Constant<Number> {
        if (value == null) {
            value = nodeA.value().dot(nodeB.value())
        }
        return value!!
    }

    override fun zeroGrad() {
        super.zeroGrad()
        nodeA.zeroGrad()
        nodeB.zeroGrad()
    }

    override fun backward(gradient: Constant<Number>) {
        this.gradient += gradient

        val consA = Constant(values = nodeA.value().values(), shape = nodeA.value().shape())
        val consB = Constant(values = nodeB.value().values(), shape = nodeB.value().shape())

        if (nodeA.transposed()) {
            nodeB.backward(diff(consA).transpose() * gradient)
            nodeA.backward(diff(consB.transpose()).transpose() * gradient)
        }

        if (nodeB.transposed()) {
            nodeA.backward(diff(consB.transpose()) * gradient)
            nodeB.backward(diff(consA) * gradient)
        }

    }

    private fun diff(cons: Constant<Number>): Constant<Number> {
        val shape = cons.shape()
        val row = shape[0]
        val col = shape[1]

        val colValues = Array(col) { i ->
            cons.getColumn(i).sumOf { it.toDouble() }
        }

        val values = Array(row * col) { colValues[it % col] }
        return Constant(values = values.map { it }, shape = shape)
    }

}
