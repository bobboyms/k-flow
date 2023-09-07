import br.com.kflow.linear.*
import java.util.*

fun calcError(a:Variable, b:Variable): Node<Number> {
    return Pow((a - b), Constant(2))
}

class Perceptron(input:Int, neurons:Int) {
    var w = Variable(values = generateNormalDistribution(neurons*input).toList(), shape = arrayOf(neurons, input), name = "a", requiresGrad = true)
//    var b = Variable(values = listOf(1.2f, 2.5f, 3.2f, 4.5f, 3.5f, 2.5f), shape = arrayOf(1, neurons), name = "a", requiresGrad = true)

    fun forward(x:Variable):Node<Number> {
        return Matmul(x,w.T())
    }

    private fun generateNormalDistribution(size: Int): DoubleArray {
        val random = Random()
        return DoubleArray(size) {
            random.nextGaussian()
        }
    }

}

fun main(args: Array<String>) {
    val variable1 = Variable(values = listOf(1.2f, 2.5f, 3.2f, 4.5f, 3.5f, 2.5f), shape = arrayOf(3, 2), name = "a", requiresGrad = true)
    val variable2 = Variable(values = listOf(5.2f, 2.3f, 3.5f, 6.5f, 8.5f, 6.5f), shape = arrayOf(3, 2), name = "b", requiresGrad = true)

//    variable1.value().exp().printMatrix()

    val c = Exp(variable1)

    c.value().printMatrix()
    c.backward(Constant(1))

    variable1.grad().printMatrix()

}
