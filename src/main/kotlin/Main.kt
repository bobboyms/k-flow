import br.com.kflow.computerGraph.*
import br.com.kflow.value.Value
import br.com.kflow.activations.*
import br.com.kflow.loss.*


fun main(args: Array<String>) {

    val nodeA = Tensor(5.0, requiresGrad = true)
    val nodeB = Tensor(3.0, requiresGrad = true)
    val multiplyNode = Multiply(nodeA, nodeB)
    multiplyNode.backward(Value(1.0))

    nodeA.grad().printMatrix()
}
