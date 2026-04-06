package com.example.embedding

import org.json.JSONObject
import java.io.File
import java.io.FileInputStream

/**
 * JVM 테스트 러너 — Android 단말 없이 토크나이저를 검증한다.
 *
 * BpeTokenizerTest.kt / UnigramTokenizerTest.kt 와 동일한 TC를
 * 콘솔에서 직접 실행. golden JSON과 1:1 비교.
 *
 * Usage:
 *   kotlinc -cp lib/json-20231013.jar *.kt -include-runtime -d test.jar
 *   java -cp "test.jar;lib/json-20231013.jar" com.example.embedding.RunTokenizerTestsKt
 */

data class TestResult(val name: String, val passed: Boolean, val detail: String = "")

fun loadGolden(path: String): Map<String, Pair<String, List<Int>>> {
    val json = JSONObject(File(path).readText(Charsets.UTF_8))
    val result = mutableMapOf<String, Pair<String, List<Int>>>()
    for (key in json.keys()) {
        val entry = json.getJSONObject(key)
        val text = entry.getString("text")
        val idsArr = entry.getJSONArray("ids")
        val ids = (0 until idsArr.length()).map { idsArr.getInt(it) }
        result[key] = text to ids
    }
    return result
}

fun runGoldenTests(
    tokenizerName: String,
    encodeFn: (String) -> IntArray,
    golden: Map<String, Pair<String, List<Int>>>,
): List<TestResult> {
    val results = mutableListOf<TestResult>()

    for ((key, pair) in golden.entries.sortedBy { it.key }) {
        val (text, expectedIds) = pair
        try {
            val actual = encodeFn(text).toList()
            if (actual == expectedIds) {
                results.add(TestResult(key, true))
            } else {
                // 차이 위치 찾기
                var diffPos = -1
                for (i in 0 until minOf(actual.size, expectedIds.size)) {
                    if (actual[i] != expectedIds[i]) { diffPos = i; break }
                }
                if (diffPos == -1) diffPos = minOf(actual.size, expectedIds.size)

                results.add(TestResult(key, false,
                    "expected(${expectedIds.size}): ${expectedIds.take(8)}${if (expectedIds.size > 8) "..." else ""}\n" +
                    "         actual  (${actual.size}): ${actual.take(8)}${if (actual.size > 8) "..." else ""}\n" +
                    "         first diff at pos $diffPos"
                ))
            }
        } catch (e: Exception) {
            results.add(TestResult(key, false, "ERROR: ${e.message}"))
        }
    }

    return results
}

fun runStructuralTests(
    tokenizerName: String,
    encodeFn: (String) -> IntArray,
    decodeFn: (IntArray) -> String,
    isBpe: Boolean,
): List<TestResult> {
    val results = mutableListOf<TestResult>()

    // Determinism
    val deterTexts = listOf(
        "Hello, nice to meet you",
        "안녕하세요, 반갑습니다",
        "Hello 😀 World 🌍",
    )
    for (text in deterTexts) {
        val first = encodeFn(text).toList()
        val second = encodeFn(text).toList()
        results.add(TestResult(
            "determinism/${text.take(20)}",
            first == second,
            if (first != second) "1st: $first\n2nd: $second" else ""
        ))
    }

    // Special token positions
    if (isBpe) {
        // EOS at end
        for (text in listOf("Hello", "안녕하세요", "Test 123")) {
            val ids = encodeFn(text)
            results.add(TestResult(
                "eos_position/$text",
                ids.isNotEmpty() && ids.last() == 41775,
                if (ids.isEmpty()) "empty" else "last=${ids.last()}"
            ))
        }
    } else {
        // CLS at start, SEP at end
        for (text in listOf("Hello", "안녕하세요", "Test 123")) {
            val ids = encodeFn(text)
            val ok = ids.size >= 3 && ids.first() == 0 && ids.last() == 2
            results.add(TestResult(
                "cls_sep_position/$text",
                ok,
                "ids(${ids.size}): [${ids.first()}, ..., ${ids.last()}]"
            ))
        }
        // Empty → [CLS, SEP]
        val emptyIds = encodeFn("")
        results.add(TestResult(
            "empty_string",
            emptyIds.size == 2 && emptyIds[0] == 0 && emptyIds[1] == 2,
            "ids: ${emptyIds.toList()}"
        ))
    }

    // AttentionMask (encode returns inputIds, but we verify length consistency)
    // Round-trip
    for (text in listOf("Hello, nice to meet you", "The quick brown fox")) {
        val encoded = encodeFn(text)
        val decoded = decodeFn(encoded)
        results.add(TestResult(
            "roundtrip/${text.take(20)}",
            decoded == text,
            if (decoded != text) "decoded='${decoded.take(50)}'" else ""
        ))
    }

    return results
}

fun printResults(tokenizerName: String, category: String, results: List<TestResult>) {
    val passed = results.count { it.passed }
    val failed = results.count { !it.passed }
    val status = if (failed == 0) "PASS" else "FAIL"

    println("  [$category] $passed/${ passed + failed } passed  [$status]")
    for (r in results.filter { !it.passed }) {
        println("    FAIL ${r.name}")
        if (r.detail.isNotEmpty()) {
            for (line in r.detail.lines()) {
                println("         $line")
            }
        }
    }
}

fun main() {
    val baseDir = System.getProperty("user.dir")
    // golden JSON 경로: 상위 디렉토리 또는 현재 디렉토리
    val goldenBpePath = listOf(
        "$baseDir/golden_bpe.json",
        "$baseDir/../golden_bpe.json",
    ).first { File(it).exists() }
    val goldenUnigramPath = listOf(
        "$baseDir/golden_unigram.json",
        "$baseDir/../golden_unigram.json",
    ).first { File(it).exists() }

    // tokenizer.json 경로: HuggingFace 캐시에서 찾기
    val cacheBase = System.getProperty("user.home") + "/.cache/huggingface/hub"

    fun findTokenizerJson(modelId: String): String {
        val safeName = "models--" + modelId.replace("/", "--")
        val snapDir = File("$cacheBase/$safeName/snapshots")
        if (snapDir.exists()) {
            for (dir in snapDir.listFiles()!!) {
                val f = File(dir, "tokenizer.json")
                if (f.exists()) return f.absolutePath
            }
        }
        throw RuntimeException("tokenizer.json not found for $modelId. Run: python -c \"from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('$modelId')\"")
    }

    var totalPass = 0
    var totalFail = 0

    // ═══════════════════════════════════════════════════════════
    // BPE Tokenizer
    // ═══════════════════════════════════════════════════════════
    println("=" .repeat(70))
    println(" BPE TOKENIZER TEST (Jina v5)")
    println("=" .repeat(70))

    try {
        val bpePath = findTokenizerJson("gomyk/jina-v5-h256-distilled-conv")
        val bpeTok = BpeTokenizer.fromInputStream(FileInputStream(bpePath))
        val goldenBpe = loadGolden(goldenBpePath)

        val categories = listOf(
            "lang_basic", "lang_long", "special_chars", "edge_cases",
            "mixed_lang", "numbers_code", "stress"
        )

        for (cat in categories) {
            val catGolden = goldenBpe.filter { it.key.startsWith("$cat/") }
            val results = runGoldenTests("BPE", { bpeTok.encode(it).inputIds }, catGolden)
            printResults("BPE", cat, results)
            totalPass += results.count { it.passed }
            totalFail += results.count { !it.passed }
        }

        // Structural tests
        val structResults = runStructuralTests("BPE",
            { bpeTok.encode(it).inputIds },
            { bpeTok.decode(it) },
            isBpe = true)
        printResults("BPE", "structural", structResults)
        totalPass += structResults.count { it.passed }
        totalFail += structResults.count { !it.passed }

    } catch (e: Exception) {
        println("  ERROR: ${e.message}")
        e.printStackTrace()
    }

    // ═══════════════════════════════════════════════════════════
    // Unigram Tokenizer
    // ═══════════════════════════════════════════════════════════
    println()
    println("=" .repeat(70))
    println(" UNIGRAM TOKENIZER TEST (mE5)")
    println("=" .repeat(70))

    try {
        val unigramPath = findTokenizerJson("gomyk/me5s-student-me5s_compressed_distilled")
        val unigramTok = UnigramTokenizer.fromInputStream(FileInputStream(unigramPath))
        val goldenUnigram = loadGolden(goldenUnigramPath)

        val categories = listOf(
            "lang_basic", "lang_long", "special_chars", "edge_cases",
            "mixed_lang", "numbers_code", "stress"
        )

        for (cat in categories) {
            val catGolden = goldenUnigram.filter { it.key.startsWith("$cat/") }
            val results = runGoldenTests("Unigram", { unigramTok.encode(it).inputIds }, catGolden)
            printResults("Unigram", cat, results)
            totalPass += results.count { it.passed }
            totalFail += results.count { !it.passed }
        }

        // Structural tests
        val structResults = runStructuralTests("Unigram",
            { unigramTok.encode(it).inputIds },
            { unigramTok.decode(it) },
            isBpe = false)
        printResults("Unigram", "structural", structResults)
        totalPass += structResults.count { it.passed }
        totalFail += structResults.count { !it.passed }

    } catch (e: Exception) {
        println("  ERROR: ${e.message}")
        e.printStackTrace()
    }

    // ═══════════════════════════════════════════════════════════
    // Summary
    // ═══════════════════════════════════════════════════════════
    println()
    println("─" .repeat(70))
    println(" TOTAL: $totalPass passed, $totalFail failed / ${totalPass + totalFail}")
    if (totalFail == 0) {
        println(" ✓ ALL ${totalPass} TESTS PASSED")
    } else {
        println(" ✗ $totalFail FAILURES")
    }
    println("─" .repeat(70))

    if (totalFail > 0) System.exit(1)
}
