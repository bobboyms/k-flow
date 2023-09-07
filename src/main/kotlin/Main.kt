import linear.*
import kotlin.math.ceil

fun calcError(a:Variable, b:Variable): Node<Number> {
    return Pow((a - b), Constant(2))
}

fun main(args: Array<String>) {
    val variable1 = Variable(values = listOf(1.2f, 2.5f, 3.2f, 4.5f, 3.5f, 2.5f), shape = arrayOf(3, 2), name = "a", requiresGrad = true)
    val variable2 = Variable(values = listOf(5.2f, 2.3f, 3.5f, 6.5f, 8.5f, 6.5f), shape = arrayOf(3, 2), name = "b", requiresGrad = true)

    val lr = Constant(0.01 as Number)

    for (step in 0..5000) {
        val c = calcError(variable1,variable2)
        val loss = Sum(c)
        loss.backward(Constant(1))

        variable1.changeValue(variable1.value() - lr * variable1.grad())
        c.zeroGrad()

        if (step % 1000 == 0) {
            println("Step: " + step + " Loss: " + loss.value().values()[0])
        }

    }

    variable1.value().printMatrix()

}
