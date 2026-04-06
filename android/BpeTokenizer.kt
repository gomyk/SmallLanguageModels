package com.example.embedding

import java.io.InputStream
import org.json.JSONObject
import org.json.JSONArray

/**
 * ByteLevel BPE Tokenizer (GPT-2/Jina v5 스타일)
 *
 * tokenizer.json 하나로 동작한다.
 * HuggingFace tokenizers 라이브러리의 동작을 코틀린으로 재현.
 *
 * Pipeline:
 *   1. Pre-tokenize: regex로 단어 단위 분리
 *   2. Byte-level encoding: 각 byte를 unicode char로 매핑
 *   3. BPE merge: merge 규칙을 반복 적용해 토큰 생성
 *   4. Vocab lookup: 토큰 문자열 → token ID
 *   5. Post-process: EOS 토큰 추가
 *
 * Usage:
 *   val tokenizer = BpeTokenizer.fromInputStream(assets.open("tokenizer.json"))
 *   val encoded = tokenizer.encode("Hello world")
 *   // encoded.inputIds = [6404, 1544, 41775]
 *   // encoded.attentionMask = [1, 1, 1]
 */
class BpeTokenizer private constructor(
    private val vocab: Map<String, Int>,       // token string → ID
    private val merges: List<Pair<String, String>>,  // BPE merge rules (순서 = 우선순위)
    private val mergeRanks: Map<Pair<String, String>, Int>,  // merge pair → rank
    private val eosTokenId: Int,
    private val padTokenId: Int,
) {
    data class Encoded(
        val inputIds: IntArray,
        val attentionMask: IntArray,
    )

    // ── Byte ↔ Unicode 매핑 (GPT-2 standard) ──────────────

    companion object {
        // GPT-2의 byte_to_unicode: 256 bytes를 충돌 없는 unicode로 매핑
        // printable ASCII + Latin-1은 그대로, 나머지는 U+0100부터 순서대로
        private val byteToUnicode: Map<Int, Char>
        private val unicodeToByte: Map<Char, Int>

        init {
            val bs = mutableListOf<Int>()
            val cs = mutableListOf<Int>()

            // printable ranges: ! ~ ¡ ¬ ® ÿ
            for (b in '!'.code..'~'.code) bs.add(b)
            for (b in '¡'.code..'¬'.code) bs.add(b)
            for (b in '®'.code..'ÿ'.code) bs.add(b)
            cs.addAll(bs)

            var n = 0
            for (b in 0..255) {
                if (b !in bs) {
                    bs.add(b)
                    cs.add(256 + n)
                    n++
                }
            }

            byteToUnicode = bs.zip(cs.map { it.toChar() }).toMap()
            unicodeToByte = byteToUnicode.entries.associate { (k, v) -> v to k }
        }

        /**
         * tokenizer.json InputStream에서 토크나이저를 로드한다.
         */
        fun fromInputStream(input: InputStream): BpeTokenizer {
            val jsonStr = input.bufferedReader(Charsets.UTF_8).readText()
            val json = JSONObject(jsonStr)
            val model = json.getJSONObject("model")

            // Vocab
            val vocabJson = model.getJSONObject("vocab")
            val vocab = mutableMapOf<String, Int>()
            for (key in vocabJson.keys()) {
                vocab[key] = vocabJson.getInt(key)
            }

            // Added tokens (special tokens)
            val addedTokens = json.getJSONArray("added_tokens")
            for (i in 0 until addedTokens.length()) {
                val at = addedTokens.getJSONObject(i)
                vocab[at.getString("content")] = at.getInt("id")
            }

            // Merges
            val mergesJson = model.getJSONArray("merges")
            val merges = mutableListOf<Pair<String, String>>()
            val mergeRanks = mutableMapOf<Pair<String, String>, Int>()
            for (i in 0 until mergesJson.length()) {
                val parts = mergesJson.getString(i).split(" ", limit = 2)
                if (parts.size == 2) {
                    val pair = parts[0] to parts[1]
                    merges.add(pair)
                    mergeRanks[pair] = i
                }
            }

            val eosId = vocab["<|end_of_text|>"] ?: 41775
            val padId = vocab["<|pad|>"] ?: 41777

            return BpeTokenizer(vocab, merges, mergeRanks, eosId, padId)
        }
    }

    // ── Pre-tokenize (Regex Split) ─────────────────────────

    // GPT-4 스타일 regex — 코틀린에서는 \p{L}, \p{N} 지원됨
    private val preTokenizePattern = Regex(
        """(?i:'s|'t|'re|'ve|'m|'ll|'d)""" +
        """|[^\r\n\p{L}\p{N}]?\p{L}+""" +
        """|\p{N}{1,3}""" +
        """| ?[^\s\p{L}\p{N}]+[\r\n]*""" +
        """|\s*[\r\n]+""" +
        """|\s+(?!\S)""" +
        """|\s+"""
    )

    /**
     * 텍스트를 pre-tokenize 단위로 분리한다.
     * "Hello world" → ["Hello", " world"]
     */
    private fun preTokenize(text: String): List<String> {
        return preTokenizePattern.findAll(text).map { it.value }.toList()
    }

    // ── Byte-Level Encoding ────────────────────────────────

    /**
     * 문자열을 UTF-8 bytes로 변환 → 각 byte를 GPT-2 unicode char로 매핑.
     * "Hello" → [72,101,108,108,111] → "Hello" (printable이라 동일)
     * " world" → [32,119,111,...] → "Ġworld" (space=0x20 → Ġ=U+0120)
     * "안녕" → UTF-8 bytes [0xEC,0x95,0x88,0xEB,0x85,0x95] → "ìķĪëħķ"
     */
    private fun byteLevelEncode(text: String): String {
        val bytes = text.toByteArray(Charsets.UTF_8)
        val sb = StringBuilder(bytes.size)
        for (b in bytes) {
            val unsigned = b.toInt() and 0xFF
            sb.append(byteToUnicode[unsigned] ?: '?')
        }
        return sb.toString()
    }

    // ── BPE ────────────────────────────────────────────────

    /**
     * BPE merge를 적용하여 토큰 리스트를 생성한다.
     *
     * 알고리즘:
     *   1. 문자열을 개별 char로 분리 ["H","e","l","l","o"]
     *   2. 인접 쌍 중 rank가 가장 낮은(=우선순위 높은) merge를 찾기
     *   3. 해당 쌍을 합치기: ["H","el","l","o"]
     *   4. merge할 쌍이 없을 때까지 반복
     *   5. 결과: ["Hello"] (vocab에 있는 단위)
     */
    private fun bpe(token: String): List<String> {
        if (token.length <= 1) return listOf(token)

        var pieces = token.map { it.toString() }.toMutableList()

        while (pieces.size > 1) {
            // 모든 인접 쌍의 rank를 확인, 가장 낮은 rank(=높은 우선순위) 찾기
            var bestRank = Int.MAX_VALUE
            var bestIdx = -1

            for (i in 0 until pieces.size - 1) {
                val pair = pieces[i] to pieces[i + 1]
                val rank = mergeRanks[pair]
                if (rank != null && rank < bestRank) {
                    bestRank = rank
                    bestIdx = i
                }
            }

            if (bestIdx == -1) break  // 더 이상 merge할 쌍 없음

            // merge 실행: pieces[bestIdx]와 pieces[bestIdx+1]을 합침
            val merged = pieces[bestIdx] + pieces[bestIdx + 1]
            pieces[bestIdx] = merged
            pieces.removeAt(bestIdx + 1)
        }

        return pieces
    }

    // ── Encode ─────────────────────────────────────────────

    /**
     * 텍스트를 token ID 배열로 변환한다.
     *
     * Pipeline:
     *   "Hello world"
     *   → pre-tokenize: ["Hello", " world"]
     *   → byte-level:   ["Hello", "Ġworld"]
     *   → BPE:          [["Hello"], ["Ġworld"]]
     *   → vocab lookup: [6404, 1544]
     *   → post-process: [6404, 1544, 41775]  (+ EOS)
     */
    fun encode(text: String): Encoded {
        val words = preTokenize(text)
        val tokenIds = mutableListOf<Int>()

        for (word in words) {
            val encoded = byteLevelEncode(word)
            val bpeTokens = bpe(encoded)

            for (token in bpeTokens) {
                val id = vocab[token]
                if (id != null) {
                    tokenIds.add(id)
                } else {
                    // Unknown token — byte-level fallback (개별 char)
                    for (ch in token) {
                        val charId = vocab[ch.toString()]
                        if (charId != null) {
                            tokenIds.add(charId)
                        }
                        // 그래도 없으면 skip (byte-level BPE에서는 거의 없음)
                    }
                }
            }
        }

        // Post-process: EOS 토큰 추가
        tokenIds.add(eosTokenId)

        val inputIds = tokenIds.toIntArray()
        val attentionMask = IntArray(inputIds.size) { 1 }

        return Encoded(inputIds, attentionMask)
    }

    /**
     * 여러 텍스트를 batch로 encode하고, 가장 긴 것에 맞춰 padding한다.
     */
    fun encodeBatch(
        texts: List<String>,
        maxLength: Int = 128,
    ): Pair<Array<IntArray>, Array<IntArray>> {
        val encoded = texts.map { encode(it) }
        val maxLen = minOf(maxLength, encoded.maxOf { it.inputIds.size })

        val batchIds = Array(texts.size) { i ->
            val ids = encoded[i].inputIds
            if (ids.size >= maxLen) {
                ids.copyOf(maxLen)
            } else {
                IntArray(maxLen).also { padded ->
                    ids.copyInto(padded)
                    for (j in ids.size until maxLen) padded[j] = padTokenId
                }
            }
        }

        val batchMask = Array(texts.size) { i ->
            val origLen = minOf(encoded[i].inputIds.size, maxLen)
            IntArray(maxLen).also { mask ->
                for (j in 0 until origLen) mask[j] = 1
            }
        }

        return batchIds to batchMask
    }

    /**
     * Token ID 배열을 텍스트로 디코딩한다.
     */
    fun decode(ids: IntArray): String {
        val idToToken = vocab.entries.associate { (k, v) -> v to k }
        val sb = StringBuilder()
        for (id in ids) {
            if (id == eosTokenId || id == padTokenId) continue
            val token = idToToken[id] ?: continue
            sb.append(token)
        }
        // Unicode → bytes → UTF-8 string
        val bytes = ByteArray(sb.length).also { arr ->
            for (i in sb.indices) {
                arr[i] = (unicodeToByte[sb[i]] ?: sb[i].code).toByte()
            }
        }
        return String(bytes, Charsets.UTF_8)
    }
}
