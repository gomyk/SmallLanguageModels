package com.example.embedding

import java.io.InputStream
import org.json.JSONObject

/**
 * Unigram (SentencePiece) Tokenizer — XLMRoberta / mE5 스타일
 *
 * tokenizer.json 하나로 동작한다.
 *
 * Jina(BPE)와의 차이:
 *   BPE:     merge 규칙을 순서대로 적용 (bottom-up)
 *   Unigram: 모든 가능한 분할 중 확률이 가장 높은 것을 선택 (top-down)
 *
 * Pipeline:
 *   1. Normalize: 연속 공백 제거
 *   2. Pre-tokenize: Metaspace (공백 → ▁ 접두사)
 *   3. Unigram tokenize: Viterbi 알고리즘으로 최적 분할
 *   4. Vocab lookup: 토큰 → ID
 *   5. Post-process: [CLS] + tokens + [SEP]
 *
 * Usage:
 *   val tokenizer = UnigramTokenizer.fromInputStream(assets.open("tokenizer.json"))
 *   val encoded = tokenizer.encode("Hello world")
 *   // encoded.inputIds = [0, 8258, 4039, 2]
 *   // encoded.attentionMask = [1, 1, 1, 1]
 */
class UnigramTokenizer private constructor(
    private val vocab: List<Pair<String, Double>>,  // (piece, log_prob) — index = token ID
    private val pieceToId: Map<String, Int>,
    private val unkId: Int,
    private val clsId: Int,    // <s> = 0
    private val sepId: Int,    // </s> = 2
    private val padId: Int,    // <pad> = 1
) {
    data class Encoded(
        val inputIds: IntArray,
        val attentionMask: IntArray,
    )

    companion object {
        // SentencePiece의 공백 표현 문자
        private const val METASPACE = '▁'  // U+2581

        fun fromInputStream(input: InputStream): UnigramTokenizer {
            val jsonStr = input.bufferedReader(Charsets.UTF_8).readText()
            val json = JSONObject(jsonStr)
            val model = json.getJSONObject("model")

            // Vocab: [[piece, score], ...] — index가 token ID
            val vocabJson = model.getJSONArray("vocab")
            val vocab = mutableListOf<Pair<String, Double>>()
            val pieceToId = mutableMapOf<String, Int>()

            for (i in 0 until vocabJson.length()) {
                val entry = vocabJson.getJSONArray(i)
                val piece = entry.getString(0)
                val score = entry.getDouble(1)
                vocab.add(piece to score)
                pieceToId[piece] = i
            }

            // Special tokens from added_tokens
            val addedTokens = json.getJSONArray("added_tokens")
            for (i in 0 until addedTokens.length()) {
                val at = addedTokens.getJSONObject(i)
                val content = at.getString("content")
                val id = at.getInt("id")
                if (id < vocab.size) {
                    // 이미 vocab에 있음
                    pieceToId[content] = id
                }
            }

            val unkId = model.optInt("unk_id", 3)
            val clsId = pieceToId["<s>"] ?: 0
            val sepId = pieceToId["</s>"] ?: 2
            val padId = pieceToId["<pad>"] ?: 1

            return UnigramTokenizer(vocab, pieceToId, unkId, clsId, sepId, padId)
        }
    }

    // ── Normalize ──────────────────────────────────────────

    /**
     * SentencePiece Precompiled normalizer:
     * 제어 문자, 탭, 개행 등을 공백으로 치환하고 연속 공백을 단일 공백으로 축소.
     */
    private fun normalize(text: String): String {
        return text
            .replace(Regex("[\t\n\r\\x0b\\x0c\u00a0\u2000-\u200b\u2028\u2029\u3000\ufeff]"), " ")
            .replace(Regex(" {2,}"), " ")
    }

    // ── Pre-tokenize (Metaspace) ───────────────────────────

    /**
     * Metaspace pre-tokenizer: 공백을 ▁로 치환하고, 문자열 앞에도 ▁를 붙인다.
     *
     * "Hello world" → "▁Hello▁world"
     * " Hello"      → "▁▁Hello"       (선행 공백도 ▁)
     *
     * prepend_scheme=always: 항상 맨 앞에 ▁ 추가
     */
    private fun preTokenize(text: String): String {
        // 1. 모든 공백을 ▁로 치환
        val replaced = text.replace(' ', METASPACE)
        // 2. 맨 앞에 ▁가 없으면 추가 (prepend_scheme=always)
        return if (replaced.startsWith(METASPACE)) replaced
               else "$METASPACE$replaced"
    }

    // ── Unigram Tokenize (Viterbi) ─────────────────────────

    /**
     * Viterbi 알고리즘으로 최적의 토큰 분할을 찾는다.
     *
     * 원리:
     *   text = "▁Hello"
     *   모든 위치 i에 대해, 가장 높은 확률로 text[0..i]를 분할하는 방법을 DP로 계산.
     *
     *   score[i] = max over j<i of { score[j] + log_prob(text[j..i]) }
     *
     *   text[j..i]가 vocab에 있으면 해당 log_prob 사용,
     *   없으면 skip (single char일 때만 UNK로 fallback).
     *
     * 예시: "▁Hello"
     *   position 0: 시작
     *   position 1: "▁" → vocab에 있음 (score = -3.93)
     *   position 2: "▁H" → 없음, "H" → 있음
     *   ...
     *   position 6: "▁Hello" → vocab에 있으면 한 번에 매칭
     *   → 최적: ["▁Hello"] (한 토큰으로 커버)
     */
    private fun tokenize(text: String): List<Int> {
        if (text.isEmpty()) return emptyList()

        val n = text.length

        // bestScore[i]: text[0..i)까지의 최적 score
        // bestPrev[i]: text[0..i)의 최적 분할에서 마지막 토큰의 시작 위치
        val bestScore = DoubleArray(n + 1) { Double.NEGATIVE_INFINITY }
        val bestPrev = IntArray(n + 1) { -1 }
        bestScore[0] = 0.0

        // 최대 토큰 길이 (너무 긴 것은 탐색 불필요)
        val maxPieceLen = 64

        for (end in 1..n) {
            // end 위치에서 끝나는 모든 가능한 토큰을 시도
            val startMin = maxOf(0, end - maxPieceLen)
            for (start in startMin until end) {
                if (bestScore[start] == Double.NEGATIVE_INFINITY) continue

                val piece = text.substring(start, end)
                val id = pieceToId[piece]

                if (id != null) {
                    val score = bestScore[start] + vocab[id].second
                    if (score > bestScore[end]) {
                        bestScore[end] = score
                        bestPrev[end] = start
                    }
                }
            }

            // 어떤 토큰도 매칭되지 않았으면 — single char를 UNK로 처리
            if (bestScore[end] == Double.NEGATIVE_INFINITY && end > 0) {
                // 한 글자씩 UNK 처리
                val prevEnd = end - 1
                if (bestScore[prevEnd] > Double.NEGATIVE_INFINITY) {
                    bestScore[end] = bestScore[prevEnd] + (-100.0)  // UNK penalty
                    bestPrev[end] = prevEnd
                }
            }
        }

        // Backtrack: 최적 분할 복원
        val tokenIds = mutableListOf<Int>()
        var pos = n
        while (pos > 0) {
            val prev = bestPrev[pos]
            if (prev < 0) {
                // fallback: 한 글자씩 UNK
                tokenIds.add(unkId)
                pos--
            } else {
                val piece = text.substring(prev, pos)
                val id = pieceToId[piece] ?: unkId
                tokenIds.add(id)
                pos = prev
            }
        }

        tokenIds.reverse()
        return tokenIds
    }

    // ── Encode ─────────────────────────────────────────────

    /**
     * 텍스트를 token ID 배열로 변환한다.
     *
     * Pipeline:
     *   "Hello world"
     *   → normalize:     "Hello world"
     *   → preTokenize:   "▁Hello▁world"
     *   → tokenize:      [8258, 4039]  (Viterbi)
     *   → post-process:  [0, 8258, 4039, 2]  (CLS + tokens + SEP)
     */
    fun encode(text: String): Encoded {
        val normalized = normalize(text)
        if (normalized.isEmpty()) {
            return Encoded(intArrayOf(clsId, sepId), intArrayOf(1, 1))
        }
        val preTokenized = preTokenize(normalized)
        val tokenIds = tokenize(preTokenized)

        // Post-process: [CLS] + tokens + [SEP]
        val ids = IntArray(tokenIds.size + 2)
        ids[0] = clsId
        for (i in tokenIds.indices) ids[i + 1] = tokenIds[i]
        ids[ids.size - 1] = sepId

        val mask = IntArray(ids.size) { 1 }
        return Encoded(ids, mask)
    }

    /**
     * 여러 텍스트를 batch로 encode하고 padding한다.
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
                    for (j in ids.size until maxLen) padded[j] = padId
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
        val sb = StringBuilder()
        for (id in ids) {
            if (id == clsId || id == sepId || id == padId) continue
            if (id == unkId) { sb.append("?"); continue }
            if (id in vocab.indices) {
                sb.append(vocab[id].first)
            }
        }
        // ▁ → 공백, 선두 공백 제거
        return sb.toString().replace(METASPACE, ' ').trimStart()
    }
}
