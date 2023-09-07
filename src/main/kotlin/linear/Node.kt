import linear.Add
import linear.Constant
import linear.Multiply
import linear.Sub

abstract class Node<T: Number> {
    protected var gradient = Constant(0.0 as T)
    abstract fun value(): Constant<T>
    abstract fun backward(gradient: Constant<T>)

    protected var value: Constant<Number>? = null
    private var transposed = false

    fun transposed():Boolean {
        return transposed
    }

    fun grad():Constant<T> {
        return gradient
    }

    fun changeValue(value: Constant<Number>) {
        this.value = value
    }

    fun T() : Node<T> {
        transposed = !transposed
        val t = value().transpose()
        value = Constant(t.values(),shape=t.shape())
        return this
    }

    open fun zeroGrad() {
        this.gradient = Constant(0.0 as T)
    }

}

// Since Node is a generic class, you have to indicate its type
operator fun Node<Number>.plus(other: Node<Number>): Node<Number> {
    return Add(this, other)
}

operator fun Node<Number>.times(other: Node<Number>): Node<Number> {
    return Multiply(this, other)
}

operator fun Node<Number>.minus(other: Node<Number>): Node<Number> {
    return Sub(this, other)
}