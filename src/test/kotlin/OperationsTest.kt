import br.com.kflow.linear.*
import br.com.kflow.value.Value
import br.com.kflow.computerGraph.*
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
        var matmulNode = Matmul(nodeA, nodeB.transpose())

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

        matmulNode = Matmul(nodeA,nodeB.transpose())

        assertEquals(listOf(5.0, 11.0, 11.0, 25.0, 61.0, 83.0, 83.0, 113.0), matmulNode.value().values())

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

        matmulNode = Matmul(nodeA.transpose(),nodeB)
        assertEquals(listOf(10.0, 14.0, 14.0, 20.0, 74.0, 86.0, 86.0, 100.0), matmulNode.value().values())

        nodeA = Tensor(values = listOf(
            1, 2, 3, 4,
            5, 6, 7, 8,
            9, 10, 11, 12,
            13, 14, 15, 16,
            17, 18, 19, 20,
            21, 22, 23, 24,
        ), shape = arrayOf(3,2,4), requiresGrad = true)

        nodeB = Tensor(values = listOf(
            1, 2, 3, 4,
            5, 6, 7, 8,
            9, 10, 11, 12,
            13, 14, 15, 16,
            17, 18, 19, 20,
            21, 22, 23, 24,
        ), shape = arrayOf(3,2,4), requiresGrad = true)

        matmulNode = Matmul(nodeA, nodeB.transpose())
        assertEquals(listOf(30.0, 70.0,70.0, 174.0,446.0, 614.0, 614.0, 846.0,1374.0, 1670.0, 1670.0, 2030.0), matmulNode.value().values())

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

        var matmul = Matmul(nodeA, nodeB.transpose())
        var result = Sum(matmul)
        result.backward(Value(1.0 as Number))

        assertEquals(listOf(17.199999809265137, 15.299999952316284, 17.199999809265137, 15.299999952316284, 17.199999809265137, 15.299999952316284), nodeA.grad().values())
        assertEquals(listOf(7.900000095367432, 9.5, 7.900000095367432, 9.5, 7.900000095367432, 9.5), nodeB.grad().values())

        nodeA = Tensor(values = listOf(1.2f, 2.5f, 3.2f, 4.5f, 3.5f, 2.5f), shape = arrayOf(3, 2), name = "a", requiresGrad = true)
        nodeB = Tensor(values = listOf(5.2f, 2.3f, 3.5f, 6.5f, 8.5f, 6.5f), shape = arrayOf(3, 2), name = "b", requiresGrad = true)

        matmul = Matmul(nodeA.transpose(), nodeB)
        result = Sum(matmul)
        result.backward(Value(1.0 as Number))

        assertEquals(listOf(7.499999761581421, 7.499999761581421,10.0, 10.0,15.0, 15.0), nodeA.grad().values())
        assertEquals(listOf(3.700000047683716, 3.700000047683716, 7.700000047683716, 7.700000047683716, 6.0, 6.0), nodeB.grad().values())

        val v1 = Tensor(values = listOf(1.2, 3.5, 3.2), shape = arrayOf(1, 3), requiresGrad = true)
        val v2 = Tensor(values = listOf(1.2, 2.5, 3.2, 4.5, 3.5, 2.5), shape = arrayOf(2, 3), requiresGrad = true)

        val mm = Matmul(v1, v2.transpose())
        val r = Sum(mm)
        r.backward(Value(1.0))

        assertEquals(listOf(5.7, 6.0, 5.7), v1.grad().values())
        assertEquals(listOf(1.2, 3.5, 3.2, 1.2, 3.5, 3.2), v2.grad().values())

        val local = Tensor(values = listOf(
            1, 2, 3, 4,
            5, 6, 7, 8,
            9, 10, 11, 12,
            13, 14, 15, 16,
            17, 18, 19, 20,
            21, 22, 23, 24,
        ), shape = arrayOf(3,2,4), requiresGrad = true)

        val other = Tensor(values = listOf(
            1, 2, 3, 4,
            5, 6, 7, 8,
            9, 10, 11, 12,
            13, 14, 15, 16,
            17, 18, 19, 20,
            21, 22, 23, 24,
        ), shape = arrayOf(3,2,4), requiresGrad = true)

        val x = Matmul(local, other.transpose())

        val n = Sum(x)
        n.backward(Value(1.0))
        assertEquals(listOf(6.0, 8.0, 10.0, 12.0,6.0, 8.0, 10.0, 12.0,22.0, 24.0, 26.0, 28.0,
            22.0, 24.0, 26.0, 28.0,
            38.0, 40.0, 42.0, 44.0,
            38.0, 40.0, 42.0, 44.0), local.grad().values())

        assertEquals(listOf(6.0, 8.0, 10.0, 12.0,6.0, 8.0, 10.0, 12.0,22.0, 24.0, 26.0, 28.0,
            22.0, 24.0, 26.0, 28.0,
            38.0, 40.0, 42.0, 44.0,
            38.0, 40.0, 42.0, 44.0), other.grad().values())

    }

    // Similar tests can be written for Dot class
}

