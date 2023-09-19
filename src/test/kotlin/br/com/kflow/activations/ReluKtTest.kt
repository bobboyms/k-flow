package br.com.kflow.activations

import br.com.kflow.computerGraph.Sum
import br.com.kflow.computerGraph.Tensor
import br.com.kflow.value.Value
import org.junit.jupiter.api.Test

import org.junit.jupiter.api.Assertions.*

class ReluKtTest {

    @Test
    fun relu() {
        val nodeA = Tensor(values = listOf(
            -3.0, -0.5, 0.0, 0.5, 1.0
        ), shape = arrayOf(1,5), requiresGrad = true)

        val r = relu(nodeA)
        assertEquals(listOf(0.0, 0.0, 0.0, 0.5, 1.0), r.value().values())

    }

    @Test
    fun derivativeRelu() {
        val nodeA = Tensor(values = listOf(
            -3.0, -0.5, 0.0, 0.5, 1.0
        ), shape = arrayOf(1,5), requiresGrad = true)

        val c = Sum(relu(nodeA))
        c.backward(Value(1))

        assertEquals(listOf(0.0, 0.0, 0.0, 1.0, 1.0), nodeA.grad().values())

    }
}