package br.com.kflow.linear

@Suppress("UNCHECKED_CAST")
fun <T : Number> Number.toT(targetType: Class<out T>): T {
    return when (targetType.name) {
        "double", java.lang.Double::class.java.name -> toDouble()
        "float", java.lang.Float::class.java.name -> toFloat()
        "long", java.lang.Long::class.java.name -> toLong()
        "int", java.lang.Integer::class.java.name -> toInt()
        "short", java.lang.Short::class.java.name -> toShort()
        "byte", java.lang.Byte::class.java.name -> toByte()
        else -> {
            println("Throwing exception, targetType is: $targetType")
            throw IllegalArgumentException("Type not supported: $targetType")
        }
    } as T
}