package br.com.kflow.loss

import br.com.kflow.linear.Constant
import br.com.kflow.linear.Variable
import org.junit.jupiter.api.Test

import org.junit.jupiter.api.Assertions.*
import kotlin.math.abs

class LossFunctionsTest {

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
    fun derivativeQuadratic() {

        val v1 = Variable(values = listOf(1.0, 2.0, 3.0), shape = arrayOf(1,3), requiresGrad = true)
        val v2 = Variable(values = listOf(1.1, 1.9, 3.2), shape = arrayOf(1,3), requiresGrad = true)

        val r = quadratic(v1,v2)
        r.backward(Constant(1))

        assertEquals(listOf(-0.20000000000000018, 0.20000000000000018, -0.40000000000000036), v1.grad().values())
        assertEquals(listOf(0.20000000000000018, -0.20000000000000018, 0.40000000000000036), v2.grad().values())
    }
    @Test
    fun MSE() {
        val v1 = Variable(values = listOf(1.0, 2.0, 3.0), shape = arrayOf(1,3))
        val v2 = Variable(values = listOf(1.1, 1.9, 3.2), shape = arrayOf(1,3))

        val r = MSE(v1, v2)
        assertEquals(0.02f, r.value().values()[0])
    }

    @Test
    fun derivativeMSE() {
        val v1 = Variable(values = listOf(1.0, 2.0, 3.0), shape = arrayOf(1,3), name = "v1", requiresGrad = true)
        val v2 = Variable(values = listOf(1.1, 1.9, 3.2), shape = arrayOf(1,3), name = "v2", requiresGrad = true)

        val r = MSE(v1, v2)
        r.backward(Constant(1.0))

        assertEquals(listOf(-0.06666666865348822, 0.06666666865348822, -0.13333333730697644), v1.grad().values())
        assertEquals(listOf(0.06666666865348822, -0.06666666865348822, 0.13333333730697644), v2.grad().values())
    }
}