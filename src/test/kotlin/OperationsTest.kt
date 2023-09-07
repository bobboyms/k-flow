import br.com.kflow.linear.*
import org.junit.jupiter.api.Assertions
import org.junit.jupiter.api.Test
import kotlin.math.abs

class NodeOperationTests {

    @Test
    fun testAdd() {
        val nodeA = Variable(5.0 as Number)
        val nodeB = Variable(3.0 as Number)
        val addNode = Add(nodeA, nodeB)

        val result = addNode.value()
        Assertions.assertEquals(Constant(8.0).values(), result.values())
    }

    @Test
    fun testSub() {
        val nodeA = Variable(values = listOf(1.2f, 2.5f, 3.2f, 4.5f, 3.5f, 2.5f), shape = arrayOf(3, 2), name = "a", requiresGrad = true)
        val nodeB = Variable(values = listOf(5.2f, 2.3f, 3.5f, 6.5f, 8.5f, 6.5f), shape = arrayOf(3, 2), name = "b", requiresGrad = true)
        val subNode = Sub(nodeA, nodeB)

        val result = subNode.value()
        assertEquals(listOf(-3.9999998, 0.20000005,-0.29999995, -2.0,-5.0, -4.0), result.values())
    }

    @Test
    fun testSum() {
        val nodeA = Variable(values = listOf(1.2f, 2.5f, 3.2f, 4.5f, 3.5f, 2.5f), shape = arrayOf(3, 2), name = "a", requiresGrad = true)
        val sumNode = Sum(nodeA)

        val result = sumNode.value()
        assertEquals(listOf(17.4), result.values())
    }

    @Test
    fun testMultiply() {
        val nodeA = Variable(5.0 as Number)
        val nodeB = Variable(3.0 as Number)
        val multiplyNode = Multiply(nodeA, nodeB)

        val result = multiplyNode.value()
        Assertions.assertEquals(Constant(15.0).values(), result.values())
    }

    @Test
    fun testPow() {
        val nodeA = Variable(values = listOf(1.2f, 2.5f, 3.2f, 4.5f, 3.5f, 2.5f), shape = arrayOf(3, 2), name = "a", requiresGrad = true)
        val result = Pow(nodeA, Constant(2))

        assertEquals(listOf(1.44, 6.25,10.240001, 20.25,12.25, 6.25), result.value().values())

    }

    private val TOLERANCE = 1e-6  // Define your own tolerance level

    fun assertEquals(expected:List<Number>, actual: List<Number>) {

        for ((e, a) in expected.zip(actual)) {
            val eDouble = e.toDouble()
            val aDouble = a.toDouble()

            assert(abs(eDouble - aDouble) <= TOLERANCE) {
                "Expected $e but got $a (difference exceeds tolerance)"
            }
        }
    }

    @Test
    fun testMatmul() {
        // Your setup code
        val nodeA = Variable(values = listOf(1.2f, 2.5f, 3.2f, 4.5f, 3.5f, 2.5f), shape = arrayOf(3, 2))
        val nodeB = Variable(values = listOf(5.2f, 2.3f, 3.5f, 6.5f, 8.5f, 6.5f), shape = arrayOf(3, 2))
        val matmulNode = Matmul(nodeA, nodeB.T())

        val result = matmulNode.value()

        val expected = listOf(11.99, 20.45, 26.45, 26.99, 40.45, 56.45, 23.949999, 28.5, 46.0)
        assertEquals(expected, result.values())

    }

    @Test
    fun testBackwardPow() {
        val nodeA = Variable(values = listOf(1.2f, 2.5f, 3.2f, 4.5f, 3.5f, 2.5f), shape = arrayOf(3, 2), name = "a", requiresGrad = true)
        var result = Pow(nodeA, Constant(2))
        result.backward(Constant(1))

        var expected = listOf(2.4, 5.0,6.4, 9.0,7.0, 5.0)
        assertEquals(expected, nodeA.grad().values())

        result.zeroGrad()
        result = Pow(nodeA, Constant(3))
        result.backward(Constant(1))

        expected = listOf(4.32, 18.75, 30.720001, 60.75, 36.75, 18.75)
        assertEquals(expected, nodeA.grad().values())

    }

    @Test
    fun testBackwardSum() {
        val nodeA = Variable(values = listOf(1.2f, 2.5f, 3.2f, 4.5f, 3.5f, 2.5f), shape = arrayOf(3, 2), name = "a", requiresGrad = true)
        val sumNode = Sum(nodeA)
        sumNode.backward(Constant(1))

        assertEquals(listOf(1,1,1,1,1,1), nodeA.grad().values())
    }

    @Test
    fun testBackwardAdd() {
        val nodeA = Variable(5.0 as Number, requiresGrad = true)
        val nodeB = Variable(3.0 as Number, requiresGrad = true)
        val addNode = Add(nodeA, nodeB)

        addNode.backward(Constant(1.0 as Number))

        Assertions.assertEquals(Constant(1.0).values(), nodeA.grad().values())
        Assertions.assertEquals(Constant(1.0).values(), nodeB.grad().values())
    }

    @Test
    fun testBackwardMultiply() {
        val nodeA = Variable(5.0 as Number, requiresGrad = true)
        val nodeB = Variable(3.0 as Number, requiresGrad = true)
        val multiplyNode = Multiply(nodeA, nodeB)
        multiplyNode.value()

        val gradient = Constant(1.0 as Number)
        multiplyNode.backward(gradient)

        Assertions.assertEquals(Constant(3.0).values(), nodeA.grad().values())
        Assertions.assertEquals(Constant(5.0).values(), nodeB.grad().values())
    }

    @Test
    fun testBackwardSub() {
        val nodeA = Variable(values = listOf(1.2f, 2.5f, 3.2f, 4.5f, 3.5f, 2.5f), shape = arrayOf(3, 2), name = "a", requiresGrad = true)
        val nodeB = Variable(values = listOf(5.2f, 2.3f, 3.5f, 6.5f, 8.5f, 6.5f), shape = arrayOf(3, 2), name = "b", requiresGrad = true)
        val subNode = Sub(nodeA, nodeB)
        subNode.backward(Constant(1))

        assertEquals(listOf(1.0, 1.0,1.0, 1.0,1.0, 1.0), nodeA.grad().values())
        assertEquals(listOf(-1.0, -1.0, -1.0, -1.0, -1.0, -1.0), nodeB.grad().values())
    }

    @Test
    fun testBackwardMatmul() {
        val nodeA = Variable(values = listOf(1.2f, 2.5f, 3.2f, 4.5f, 3.5f, 2.5f), shape = arrayOf(3, 2), name = "a", requiresGrad = true)
        val nodeB = Variable(values = listOf(5.2f, 2.3f, 3.5f, 6.5f, 8.5f, 6.5f), shape = arrayOf(3, 2), name = "b", requiresGrad = true)

        var matmul = Matmul(nodeA, nodeB.T())
        matmul.backward(Constant(1.0 as Number))

        assertEquals(listOf(17.199999809265137, 15.299999952316284, 17.199999809265137, 15.299999952316284, 17.199999809265137, 15.299999952316284), nodeA.grad().values())
        assertEquals(listOf(7.900000095367432, 9.5, 7.900000095367432, 9.5, 7.900000095367432, 9.5), nodeB.grad().values())

        nodeB.T()
        matmul.zeroGrad()
        matmul = Matmul(nodeA.T(), nodeB)
        matmul.backward(Constant(1.0 as Number))

        assertEquals(listOf(7.499999761581421, 7.499999761581421,10.0, 10.0,15.0, 15.0), nodeA.grad().values())
        assertEquals(listOf(3.700000047683716, 3.700000047683716, 7.700000047683716, 7.700000047683716, 6.0, 6.0), nodeB.grad().values())

    }

    // Similar tests can be written for Dot class
}

