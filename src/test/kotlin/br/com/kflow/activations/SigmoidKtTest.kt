package br.com.kflow.activations

import br.com.kflow.value.Value
import br.com.kflow.computerGraph.*
import org.junit.jupiter.api.Test
import kotlin.math.abs

class SigmoidKtTest {
    @Test
    fun sigmoid() {
        val tensor1 = Tensor(values = listOf(1.2f, 2.5f, 3.2f, 4.5f, 3.5f, 2.5f), shape = arrayOf(3, 2), name = "a", requiresGrad = true)
        val r = sigmoid(tensor1)

        assertEquals(listOf(
            1.3011942, 1.082085,
            1.0407622, 1.011109,
            1.0301974, 1.082085
        ), r.value().values())
    }

    @Test
    fun sigmoidDerivative() {
        val tensor1 = Tensor(values = listOf(1.2f, 2.5f, 3.2f, 4.5f, 3.5f, 2.5f), shape = arrayOf(3, 2), name = "a", requiresGrad = true)
        val r = sigmoid(tensor1)
        r.backward(Value(1))

        assertEquals(listOf(
            -0.3011941909790039, -0.08208499848842621,
            -0.04076220095157623, -0.011108996346592903,
            -0.03019738383591175, -0.08208499848842621
        ), tensor1.grad().values())
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
}