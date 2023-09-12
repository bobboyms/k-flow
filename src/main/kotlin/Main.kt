import br.com.kflow.activations.sigmoid
import br.com.kflow.linear.*
import br.com.kflow.loss.MSE
import br.com.kflow.loss.quadratic
import java.util.*


class Perceptron(input: Int, neurons: Int) {
    private var w = Variable(values = generateNormalDistribution(neurons*input).toList(), shape = arrayOf(neurons, input), name = "w", requiresGrad = true)
//    var b = Variable(values = listOf(1.2f, 2.5f, 3.2f, 4.5f, 3.5f, 2.5f), shape = arrayOf(1, neurons), name = "a", requiresGrad = true)

    fun forward(x:Node<Number>, activation: (variable: Node<Number>) -> Node<Number>):Node<Number> {
        return activation(Matmul(x,w.T()))
    }

    fun forward(x:Node<Number>):Node<Number> {
        return Matmul(x,w.T())
    }

    fun w():Node<Number> {
        return w
    }

    fun changeW(values:Constant<Number>){
        w.changeValue(values)
    }

    private fun generateNormalDistribution(size: Int): DoubleArray {
        val random = Random()
        return DoubleArray(size) {
            random.nextGaussian()
        }
    }

}

fun main(args: Array<String>) {

    val layer1 = Perceptron(3,5)
    val layer2 = Perceptron(5,30)
    val layer3 = Perceptron(30,3)

    val lr = Constant(0.001 as Number)

    for (i in 0..500+1) {
        var r = layer1.forward(Variable(values = listOf(1f, 2f, 3f), shape = arrayOf(1,3)))
        r = layer2.forward(r)
        r = layer3.forward(r)

        val loss = MSE(Variable(values = listOf(9.15, 20.0246118, 3.038389), shape = arrayOf(1,3)), r)

        if (i % 100 == 0) {
            loss.value().printMatrix()
        }

        loss.backward(Constant(1.0))

        layer1.changeW(layer1.w().value() - lr * layer1.w().grad().transpose())
        layer2.changeW(layer2.w().value() - lr * layer2.w().grad().transpose())
        loss.zeroGrad()

    }

    var r = layer1.forward(Variable(values = listOf(1f, 2f, 3f), shape = arrayOf(1,3)))
    r = layer2.forward(r)
    r = layer3.forward(r)

    r.value().printMatrix()

}
