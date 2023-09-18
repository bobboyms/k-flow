import br.com.kflow.linear.*
import br.com.kflow.value.Value
import org.junit.jupiter.api.Assertions
import org.junit.jupiter.api.Test
import kotlin.math.abs

class NodeOperationTests {

    @Test
    fun testAdd() {
        val nodeA = Tensor(5.0 as Number)
        val nodeB = Tensor(3.0 as Number)
        val addNode = Add(nodeA, nodeB)

        val result = addNode.value()
        Assertions.assertEquals(Value(8.0).values(), result.values())
    }

    @Test
    fun testSub() {
        val nodeA = Tensor(values = listOf(1.2f, 2.5f, 3.2f, 4.5f, 3.5f, 2.5f), shape = arrayOf(3, 2), name = "a", requiresGrad = true)
        val nodeB = Tensor(values = listOf(5.2f, 2.3f, 3.5f, 6.5f, 8.5f, 6.5f), shape = arrayOf(3, 2), name = "b", requiresGrad = true)
        val subNode = Sub(nodeA, nodeB)

        val result = subNode.value()
        assertEquals(listOf(-3.9999998, 0.20000005,-0.29999995, -2.0,-5.0, -4.0), result.values())
    }

    @Test
    fun testUnaryMinus() {
        val nodeA = Tensor(values = listOf(1.2f, 2.5f, 3.2f, 4.5f, 3.5f, 2.5f), shape = arrayOf(3, 2), name = "a", requiresGrad = true)
        val unaryMinusNode = UnaryMinus(nodeA)

        assertEquals(listOf(-1.2f, -2.5f, -3.2f, -4.5f, -3.5f, -2.5f), unaryMinusNode.value().values())
    }
    @Test
    fun testExp() {
        val nodeA = Tensor(values = listOf(5.2f, 2.3f, 3.5f, 6.5f, 8.5f, 6.5f), shape = arrayOf(3, 2), name = "a", requiresGrad = true)
        val expNode = Exp(nodeA)

        val result = expNode.value()
        assertEquals(listOf(181.2722, 9.974182, 33.11545, 665.14166, 4914.769, 665.14166), result.values())
    }

    @Test
    fun testSum() {
        val nodeA = Tensor(values = listOf(1.2f, 2.5f, 3.2f, 4.5f, 3.5f, 2.5f), shape = arrayOf(3, 2), name = "a", requiresGrad = true)
        val sumNode = Sum(nodeA)

        val result = sumNode.value()
        assertEquals(listOf(17.4), result.values())
    }

    @Test
    fun testMultiply() {
        val nodeA = Tensor(5.0 as Number)
        val nodeB = Tensor(3.0 as Number)
        val multiplyNode = Multiply(nodeA, nodeB)

        val result = multiplyNode.value()
        Assertions.assertEquals(Value(15.0).values(), result.values())
    }

    @Test
    fun testPow() {
        val nodeA = Tensor(values = listOf(1.2f, 2.5f, 3.2f, 4.5f, 3.5f, 2.5f), shape = arrayOf(3, 2), name = "a", requiresGrad = true)
        val result = Pow(nodeA, Value(2))

        assertEquals(listOf(1.44, 6.25,10.240001, 20.25,12.25, 6.25), result.value().values())

    }

    private val TOLERANCE = 1e-4  // Define your own tolerance level

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
        var nodeA = Tensor(values = listOf(1.2f, 2.5f, 3.2f, 4.5f, 3.5f, 2.5f), shape = arrayOf(3, 2))
        var nodeB = Tensor(values = listOf(5.2f, 2.3f, 3.5f, 6.5f, 8.5f, 6.5f), shape = arrayOf(3, 2))
        var matmulNode = Matmul(nodeA, nodeB.T())

        val result = matmulNode.value()

        val expected = listOf(11.99, 20.45, 26.45, 26.99, 40.45, 56.45, 23.949999, 28.5, 46.0)
        assertEquals(expected, result.values())

        nodeA = Tensor(values = listOf(
            1, 2,
            3, 4,
            5, 6,
            7, 8,
            ), shape = arrayOf(2,2,2), requiresGrad = true)

        nodeB = Tensor(values = listOf(
            1, 2,
            3, 4,
            5, 6,
            7, 8,
        ), shape = arrayOf(2,2,2), requiresGrad = true)


        matmulNode = Matmul(nodeA,nodeB)
        assertEquals(listOf(7, 10,15, 22,67, 78,91, 106), matmulNode.value().values())

        nodeA = Tensor(values = listOf(
            1, 2,
            3, 4,
            5, 6,
            7, 8,
        ), shape = arrayOf(2,2,2), requiresGrad = true)

        nodeB = Tensor(values = listOf(
            1, 2,
            3, 4,
            5, 6,
            7, 8,
        ), shape = arrayOf(2,2,2), requiresGrad = true)

        matmulNode = Matmul(nodeA,nodeB.T())
        assertEquals(listOf(7.0, 19.0,15.0, 43.0,34.0, 78.0,46.0, 106.0), matmulNode.value().values())

        nodeA = Tensor(values = listOf(
            1, 2,
            3, 4,
            5, 6,
            7, 8,
        ), shape = arrayOf(2,2,2), requiresGrad = true)

        nodeB = Tensor(values = listOf(
            1, 2,
            3, 4,
            5, 6,
            7, 8,
        ), shape = arrayOf(2,2,2), requiresGrad = true)

        matmulNode = Matmul(nodeA.T(),nodeB)
        assertEquals(listOf(16,22,24,34,52,60,76,88), matmulNode.value().values())

    }

    @Test
    fun testBackwardPow() {
        val nodeA = Tensor(values = listOf(1.2f, 2.5f, 3.2f, 4.5f, 3.5f, 2.5f), shape = arrayOf(3, 2), name = "a", requiresGrad = true)
        var result = Pow(nodeA, Value(2))
        result.backward(Value(1))

        var expected = listOf(2.4, 5.0,6.4, 9.0,7.0, 5.0)
        assertEquals(expected, nodeA.grad().values())

        result.zeroGrad()
        result = Pow(nodeA, Value(3))
        result.backward(Value(1))

        expected = listOf(4.32, 18.75, 30.720001, 60.75, 36.75, 18.75)
        assertEquals(expected, nodeA.grad().values())

    }

    @Test
    fun testBackwardUnaryMinus() {
        val nodeA = Tensor(values = listOf(1.2f, 2.5f, 3.2f, 4.5f, 3.5f, 2.5f), shape = arrayOf(3, 2), name = "a", requiresGrad = true)
        val unaryMinusNode = UnaryMinus(nodeA)

        unaryMinusNode.backward(Value(1))
        assertEquals(listOf(-1,-1,-1,-1,-1,-1), nodeA.grad().values())
    }

    @Test
    fun testBackwardSum() {
        val nodeA = Tensor(values = listOf(1.2f, 2.5f, 3.2f, 4.5f, 3.5f, 2.5f), shape = arrayOf(3, 2), name = "a", requiresGrad = true)
        val sumNode = Sum(nodeA)
        sumNode.backward(Value(1))

        assertEquals(listOf(1,1,1,1,1,1), nodeA.grad().values())
    }

    @Test
    fun testBackwardExp() {
        val nodeA = Tensor(values = listOf(5.2f, 2.3f, 3.5f, 6.5f, 8.5f, 6.5f), shape = arrayOf(3, 2), name = "a", requiresGrad = true)
        val expNode = Exp(nodeA)
        expNode.backward(Value(1))
        assertEquals(listOf(181.2722, 9.974182, 33.11545, 665.14166, 4914.769, 665.14166), nodeA.grad().values())
    }

    @Test
    fun testBackwardAdd() {
        val nodeA = Tensor(5.0 as Number, requiresGrad = true)
        val nodeB = Tensor(3.0 as Number, requiresGrad = true)
        val addNode = Add(nodeA, nodeB)

        addNode.backward(Value(1.0 as Number))

        Assertions.assertEquals(Value(1.0).values(), nodeA.grad().values())
        Assertions.assertEquals(Value(1.0).values(), nodeB.grad().values())
    }

    @Test
    fun testBackwardMultiply() {
        val nodeA = Tensor(5.0 as Number, requiresGrad = true)
        val nodeB = Tensor(3.0 as Number, requiresGrad = true)
        val multiplyNode = Multiply(nodeA, nodeB)
        multiplyNode.value()

        val gradient = Value(1.0 as Number)
        multiplyNode.backward(gradient)

        Assertions.assertEquals(Value(3.0).values(), nodeA.grad().values())
        Assertions.assertEquals(Value(5.0).values(), nodeB.grad().values())
    }

    @Test
    fun testBackwardSub() {
        val nodeA = Tensor(values = listOf(1.2f, 2.5f, 3.2f, 4.5f, 3.5f, 2.5f), shape = arrayOf(3, 2), name = "a", requiresGrad = true)
        val nodeB = Tensor(values = listOf(5.2f, 2.3f, 3.5f, 6.5f, 8.5f, 6.5f), shape = arrayOf(3, 2), name = "b", requiresGrad = true)
        val subNode = Sub(nodeA, nodeB)
        subNode.backward(Value(1))

        assertEquals(listOf(1.0, 1.0,1.0, 1.0,1.0, 1.0), nodeA.grad().values())
        assertEquals(listOf(-1.0, -1.0, -1.0, -1.0, -1.0, -1.0), nodeB.grad().values())
    }

    @Test
    fun testBackwardMatmul() {
        var nodeA = Tensor(values = listOf(1.2f, 2.5f, 3.2f, 4.5f, 3.5f, 2.5f), shape = arrayOf(3, 2), name = "a", requiresGrad = true)
        var nodeB = Tensor(values = listOf(5.2f, 2.3f, 3.5f, 6.5f, 8.5f, 6.5f), shape = arrayOf(3, 2), name = "b", requiresGrad = true)

        var matmul = Matmul(nodeA, nodeB.T())
        var result = Sum(matmul)
        result.backward(Value(1.0 as Number))

        assertEquals(listOf(17.199999809265137, 15.299999952316284, 17.199999809265137, 15.299999952316284, 17.199999809265137, 15.299999952316284), nodeA.grad().values())
        assertEquals(listOf(7.900000095367432, 9.5, 7.900000095367432, 9.5, 7.900000095367432, 9.5), nodeB.grad().values())

        nodeA = Tensor(values = listOf(1.2f, 2.5f, 3.2f, 4.5f, 3.5f, 2.5f), shape = arrayOf(3, 2), name = "a", requiresGrad = true)
        nodeB = Tensor(values = listOf(5.2f, 2.3f, 3.5f, 6.5f, 8.5f, 6.5f), shape = arrayOf(3, 2), name = "b", requiresGrad = true)

        matmul = Matmul(nodeA.T(), nodeB)
        result = Sum(matmul)
        result.backward(Value(1.0 as Number))

        assertEquals(listOf(7.499999761581421, 7.499999761581421,10.0, 10.0,15.0, 15.0), nodeA.grad().values())
        assertEquals(listOf(3.700000047683716, 3.700000047683716, 7.700000047683716, 7.700000047683716, 6.0, 6.0), nodeB.grad().values())

        val v1 = Tensor(values = listOf(1.2, 3.5, 3.2), shape = arrayOf(1, 3), requiresGrad = true)
        val v2 = Tensor(values = listOf(1.2, 2.5, 3.2, 4.5, 3.5, 2.5), shape = arrayOf(2, 3), requiresGrad = true)

        val mm = Matmul(v1, v2.T())
        val r = Sum(mm)
        r.backward(Value(1.0))

        assertEquals(listOf(5.7, 6.0, 5.7), v1.grad().values())
        assertEquals(listOf(1.2, 3.5, 3.2, 1.2, 3.5, 3.2), v2.grad().values())


    }

    // Similar tests can be written for Dot class
}

