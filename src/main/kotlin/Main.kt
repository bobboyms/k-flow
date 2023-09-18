import br.com.kflow.linear.*
import br.com.kflow.value.Value
import java.util.*


class Dense(input: Int, neurons: Int) {

    private var w = Tensor(values = generateNormalDistribution(neurons*input).toList(), shape = arrayOf(neurons, input), name = "w", requiresGrad = true)
//    var b = Variable(values = listOf(1.2f, 2.5f, 3.2f, 4.5f, 3.5f, 2.5f), shape = arrayOf(1, neurons), name = "a", requiresGrad = true)

    fun forward(x:Node<Number>, activation: (variable: Node<Number>) -> Node<Number>):Node<Number> {
        return activation(Matmul(x,w.transpose()))
    }

    fun forward(x:Node<Number>):Node<Number> {
        return Matmul(x,w.transpose())
    }

    fun w():Node<Number> {
        return w
    }

    fun changeW(values: Value<Number>){
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

    val local = Tensor(values = listOf(
        1, 2,
        3, 4,
        5, 6,
        7, 8,
    ), shape = arrayOf(2,2,2), requiresGrad = true)

    val other = Tensor(values = listOf(
        1, 2,
        3, 4,
        5, 6,
        7, 8,
    ), shape = arrayOf(2,2,2), requiresGrad = true)

    val x = Matmul(local, other)
    val n = Sum(x)
    n.backward(Value(1.0))

    println("local")
    local.grad().printMatrix()
    println("other")
    other.grad().printMatrix()

//    [[[ 6., 10.],
//        [ 6., 10.]],
//
//        [[ 8., 12.],
//            [ 8., 12.]]])
//    xxxx
//    tensor([[[ 4., 12.],
//        [ 6., 14.]],
//
//        [[ 4., 12.],
//            [ 6., 14.]]]


//    val n = x.dot(y)
//    n.printMatrix()

//    y.printMatrix()

//    x.printMatrix()

//    val n = x.dot(y)
//    n.printMatrix()
//    val r = Value(values = n.values(), shape = arrayOf(1,3,3))
//    val r = Value(values = n.values(), shape = arrayOf(3,3,1))
//    r.printMatrix()


//    val yValues = Variable(values = listOf(
//        14.691262, -7.3010964, 15.17082, 25.263512, -13.4990551,
//        12.691262, 20.3010964, 15.17082, 15.263512, -3.4990551,
//        11.691262, 7.3010964, 1.17082, 15.263512, 25.4990551,
//        5.691262, -7.3010964, 15.17142, 15.263512, 3.4990551,
//        12.695262, 7.3010964, 18.17082, 15.263512, -3.4990551,
//    ), shape = arrayOf(5,5))
//
//    val layer1 = Dense(3,5)
////    val layer2 = Dense(10,5)
////    val layer3 = Dense(3,5)
//
//    val lr = Value(0.1 as Number)
//
//    for (i in 0..10000) {
//
//        var r = layer1.forward(xValues)
////        r = layer2.forward(r)
////        r = sigmoid(r)
////        r = layer3.forward(r)
//
//        val loss = MSE(yValues, r)
//
//        if (i % 1000 == 0) {
//            loss.value().printMatrix()
//        }
//
//        loss.backward(Value(1.0))
//
//        layer1.changeW(layer1.w().value() - lr * layer1.w().grad().transpose())
////        layer2.changeW(layer2.w().value() - lr * layer2.w().grad().transpose())
////        layer3.changeW(layer3.w().value() - lr * layer3.w().grad().transpose())
//        loss.zeroGrad()
//
//    }
}
