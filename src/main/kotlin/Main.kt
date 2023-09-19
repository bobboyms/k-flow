import br.com.kflow.activations.sigmoid
import br.com.kflow.computerGraph.*
import br.com.kflow.value.Value
import br.com.kflow.loss.*
import br.com.kflow.nn.*
import br.com.kflow.activations.relu



fun main(args: Array<String>) {

    val xValues = Tensor(values = listOf(
        -3.0, -0.5, 0.0, 0.5, 1.0
    ), shape = arrayOf(1,5), requiresGrad = true)

    val r = relu(xValues)
    r.value().printMatrix()
    val c = Sum(r)
    c.backward(Value(1))

    xValues.grad().printMatrix()

}
