package br.com.kflow.nn

import br.com.kflow.activations.relu
import br.com.kflow.activations.sigmoid
import br.com.kflow.computerGraph.*
import br.com.kflow.loss.MSE
import br.com.kflow.loss.quadratic
import br.com.kflow.value.Value
import org.junit.jupiter.api.Test

import org.junit.jupiter.api.Assertions.*
import kotlin.math.abs

class DenseTest {

    @Test
    fun forwardActivation() {
        val xValues = Tensor(values = listOf(
            0.691262, 0.3010964, 0.17082,
        ), shape = arrayOf(1,3))

        val yValues = Tensor(values = listOf(
            1.691262
        ), shape = arrayOf(1,1))

        //Create dense layer
        val layer1 = Dense(3,1)
        layer1.changeB(Value(-1.0))
        layer1.changeW(Value(values = arrayListOf(1.0, 0.191694570151395, -0.6083775574164711), shape = arrayOf(1,3)))

        //Set learning rate
        val lr = Value(0.01 as Number)

        //start training
        for (i in 0..501) {
            val r = layer1.forward(xValues,::sigmoid)
            val loss = MSE(yValues, r)
            loss.backward(Value(1.0))

            //Adjust de Weigths and Biases
            layer1.changeW(layer1.w().value() - lr * layer1.w().grad().transpose())
            layer1.changeB(layer1.b().value() - lr * layer1.b().grad())
            loss.zeroGrad()
        }

        //Evaluate de NN
        val r = layer1.forward(xValues,::sigmoid)
        val loss = MSE(yValues, r)
        val tolerance = 1e-10
        val computedValue = loss.value().values()[0].toDouble()
        val targetValue = 8.583042E-9

        assert(abs(computedValue - targetValue) <= tolerance) {
            "Difference exceeds tolerance"
        }
    }

    @Test
    fun testForward() {

        val xValues = Tensor(values = listOf(
            0.0, 0.0,
            0.0, 1.0,
            1.0, 0.0,
            1.0, 1.0,
        ), shape = arrayOf(4,2))

        val yValues = Tensor(values = listOf(
            0.0,
            1.0,
            1.0,
            0.0
        ), shape = arrayOf(4,1))

        val layer1 = Dense(2,7)
        val layer2 = Dense(7,3)
        val layer3 = Dense(3,1)

        //Set learning rate
        val lr = Value(0.01 as Number)

        //Start training
        val epochs = 100
        for (epoch in 0..<epochs) {
            //forward
            val output1 = layer1.forward(xValues, ::relu)
            val output2 = layer2.forward(output1, ::relu)
            val output3 = layer3.forward(output2)

            //calc the error
            val loss = Sum(quadratic(yValues, output3))

            //Calc the gradient
            loss.backward(Value(1))

            //Change de Biases and Weights
            layer1.changeW(layer1.w().value() - lr * layer1.w().grad().transpose())
            layer1.changeB(layer1.b().value() - lr * layer1.b().grad())

            layer2.changeW(layer2.w().value() - lr * layer2.w().grad().transpose())
            layer2.changeB(layer2.b().value() - lr * layer2.b().grad())

            layer3.changeW(layer3.w().value() - lr * layer3.w().grad().transpose())
            layer3.changeB(layer3.b().value() - lr * layer3.b().grad())

            loss.zeroGrad()
        }

        //Evaluate
        val output1 = layer1.forward(xValues, ::relu)
        val output2 = layer2.forward(output1, ::relu)
        val output3 = layer3.forward(output2)

        val loss = MSE(yValues, output3)
        val tolerance = 1e-10
        val computedValue = loss.value().values()[0].toDouble()
        val targetValue = 2.7003776E-5

        assert(abs(computedValue - targetValue) <= tolerance) {
            "Difference exceeds tolerance"
        }

    }
}