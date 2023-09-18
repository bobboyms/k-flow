package br.com.kflow.value

interface DNarray {
    fun transpose(): DNarray
    fun shape():Array<Int>
    fun matmul(other: DNarray): DNarray
    fun pow(power: DNarray): DNarray
    fun exp(): DNarray
    fun size():Int
    fun values(): List<*>
}