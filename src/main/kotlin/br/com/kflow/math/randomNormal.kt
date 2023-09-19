package br.com.kflow.math

import java.util.*

fun randomNormal(size: Int, mean: Double = 0.0, stddev: Double = 1.0, seed: Long? = null): DoubleArray {
    val random = if (seed != null) Random(seed) else Random()

    return DoubleArray(size) {
        mean + stddev * random.nextGaussian()
    }
}