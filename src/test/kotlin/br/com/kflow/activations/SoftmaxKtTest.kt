package br.com.kflow.activations

import br.com.kflow.computerGraph.Sum
import br.com.kflow.computerGraph.Tensor
import br.com.kflow.value.Value
import org.junit.jupiter.api.Test

import org.junit.jupiter.api.Assertions.*

class SoftmaxKtTest {

    @Test
    fun softmax() {
        val xValues = Tensor(values = listOf(
            1.5, 0.8, -1.2, 1.5, 0.78
        ), shape = arrayOf(1,5), requiresGrad = true)

        val loss = Sum(softmax(xValues))
        assertEquals(listOf(1.0), loss.value().values())

//        loss.value().printMatrix()
//        loss.backward(Value(1.0))
//
//        xValues.grad().printMatrix()
    }

    @Test
    fun derivativeSoftmax() {
        val xValues = Tensor(values = listOf(
            1.5, 0.8, -1.2, 1.5, 0.78
        ), shape = arrayOf(1,5), requiresGrad = true)

        val loss = Sum(softmax(xValues))
        loss.backward(Value(1.0))
        assertEquals(listOf(0.0, 0.0, 0.0, 0.0, 0.0), xValues.grad().values())

    }
}