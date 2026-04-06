package com.example.embedding

import android.content.Context
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith

/**
 * BpeTokenizer Android Instrumented Test
 *
 * 210개 TC 중 BPE 105개를 실제 단말에서 검증한다.
 * golden_bpe.json (HuggingFace 기준 expected IDs)을 assets에서 로드하여
 * BpeTokenizer.encode() 결과와 1:1 비교.
 *
 * Setup:
 *   1. tokenizer.json → app/src/androidTest/assets/bpe_tokenizer.json
 *   2. golden_bpe.json → app/src/androidTest/assets/golden_bpe.json
 *
 * Run:
 *   ./gradlew connectedAndroidTest --tests "*.BpeTokenizerTest"
 */
@RunWith(AndroidJUnit4::class)
class BpeTokenizerTest {

    private lateinit var tokenizer: BpeTokenizer
    private lateinit var golden: Map<String, Pair<String, List<Int>>>  // key → (text, expectedIds)

    @Before
    fun setUp() {
        val ctx = InstrumentationRegistry.getInstrumentation().targetContext

        // Load tokenizer
        tokenizer = ctx.assets.open("bpe_tokenizer.json").use {
            BpeTokenizer.fromInputStream(it)
        }

        // Load golden expected IDs
        golden = loadGolden(ctx, "golden_bpe.json")
    }

    private fun loadGolden(ctx: Context, fileName: String): Map<String, Pair<String, List<Int>>> {
        val jsonStr = ctx.assets.open(fileName).bufferedReader(Charsets.UTF_8).readText()
        val json = JSONObject(jsonStr)
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

    // ═══════════════════════════════════════════════════════════
    // Helper
    // ═══════════════════════════════════════════════════════════

    private fun assertEncode(category: String) {
        val failures = mutableListOf<String>()
        var count = 0

        for ((key, pair) in golden) {
            if (!key.startsWith("$category/")) continue
            count++
            val (text, expectedIds) = pair
            val actual = tokenizer.encode(text).inputIds.toList()
            if (actual != expectedIds) {
                val label = key.substringAfter("/")
                failures.add(
                    "[$label] text=${text.take(50)}\n" +
                    "  expected(${expectedIds.size}): ${expectedIds.take(10)}${if (expectedIds.size > 10) "..." else ""}\n" +
                    "  actual  (${actual.size}): ${actual.take(10)}${if (actual.size > 10) "..." else ""}"
                )
            }
        }

        assertTrue(
            "Category [$category] not found in golden data",
            count > 0
        )

        if (failures.isNotEmpty()) {
            fail("BPE [$category] ${failures.size}/$count FAILED:\n${failures.joinToString("\n")}")
        }
    }

    // ═══════════════════════════════════════════════════════════
    // 1. 16개국어 기본 문장 (16 TC)
    // ═══════════════════════════════════════════════════════════

    @Test fun langBasic_ko() = assertSingle("lang_basic/ko")
    @Test fun langBasic_en() = assertSingle("lang_basic/en")
    @Test fun langBasic_ja() = assertSingle("lang_basic/ja")
    @Test fun langBasic_zh() = assertSingle("lang_basic/zh")
    @Test fun langBasic_es() = assertSingle("lang_basic/es")
    @Test fun langBasic_fr() = assertSingle("lang_basic/fr")
    @Test fun langBasic_de() = assertSingle("lang_basic/de")
    @Test fun langBasic_pt() = assertSingle("lang_basic/pt")
    @Test fun langBasic_it() = assertSingle("lang_basic/it")
    @Test fun langBasic_ru() = assertSingle("lang_basic/ru")
    @Test fun langBasic_ar() = assertSingle("lang_basic/ar")
    @Test fun langBasic_hi() = assertSingle("lang_basic/hi")
    @Test fun langBasic_th() = assertSingle("lang_basic/th")
    @Test fun langBasic_vi() = assertSingle("lang_basic/vi")
    @Test fun langBasic_id() = assertSingle("lang_basic/id")
    @Test fun langBasic_pl() = assertSingle("lang_basic/pl")

    // ═══════════════════════════════════════════════════════════
    // 2. 16개국어 긴 문장 (16 TC)
    // ═══════════════════════════════════════════════════════════

    @Test fun langLong_ko() = assertSingle("lang_long/ko")
    @Test fun langLong_en() = assertSingle("lang_long/en")
    @Test fun langLong_ja() = assertSingle("lang_long/ja")
    @Test fun langLong_zh() = assertSingle("lang_long/zh")
    @Test fun langLong_es() = assertSingle("lang_long/es")
    @Test fun langLong_fr() = assertSingle("lang_long/fr")
    @Test fun langLong_de() = assertSingle("lang_long/de")
    @Test fun langLong_pt() = assertSingle("lang_long/pt")
    @Test fun langLong_it() = assertSingle("lang_long/it")
    @Test fun langLong_ru() = assertSingle("lang_long/ru")
    @Test fun langLong_ar() = assertSingle("lang_long/ar")
    @Test fun langLong_hi() = assertSingle("lang_long/hi")
    @Test fun langLong_th() = assertSingle("lang_long/th")
    @Test fun langLong_vi() = assertSingle("lang_long/vi")
    @Test fun langLong_id() = assertSingle("lang_long/id")
    @Test fun langLong_pl() = assertSingle("lang_long/pl")

    // ═══════════════════════════════════════════════════════════
    // 3. 특수문자 / 이모지 / 구두점 (17 TC)
    // ═══════════════════════════════════════════════════════════

    @Test fun special_punctuationBasic() = assertSingle("special_chars/punctuation_basic")
    @Test fun special_punctuationHeavy() = assertSingle("special_chars/punctuation_heavy")
    @Test fun special_brackets() = assertSingle("special_chars/brackets")
    @Test fun special_mathSymbols() = assertSingle("special_chars/math_symbols")
    @Test fun special_currency() = assertSingle("special_chars/currency")
    @Test fun special_quotes() = assertSingle("special_chars/quotes")
    @Test fun special_dashesHyphens() = assertSingle("special_chars/dashes_hyphens")
    @Test fun special_slashesPipes() = assertSingle("special_chars/slashes_pipes")
    @Test fun special_atHash() = assertSingle("special_chars/at_hash")
    @Test fun special_emojiBasic() = assertSingle("special_chars/emoji_basic")
    @Test fun special_emojiComplex() = assertSingle("special_chars/emoji_complex")
    @Test fun special_unicodeArrows() = assertSingle("special_chars/unicode_arrows")
    @Test fun special_unicodeMath() = assertSingle("special_chars/unicode_math")
    @Test fun special_fullwidth() = assertSingle("special_chars/fullwidth")
    @Test fun special_cjkPunctuation() = assertSingle("special_chars/cjk_punctuation")
    @Test fun special_arabicPunctuation() = assertSingle("special_chars/arabic_punctuation")
    @Test fun special_mixedScriptsPunct() = assertSingle("special_chars/mixed_scripts_punct")

    // ═══════════════════════════════════════════════════════════
    // 4. 엣지 케이스 (20 TC)
    // ═══════════════════════════════════════════════════════════

    @Test fun edge_empty() = assertSingle("edge_cases/empty")
    @Test fun edge_singleChar() = assertSingle("edge_cases/single_char")
    @Test fun edge_singleSpace() = assertSingle("edge_cases/single_space")
    @Test fun edge_multipleSpaces() = assertSingle("edge_cases/multiple_spaces")
    @Test fun edge_tab() = assertSingle("edge_cases/tab")
    @Test fun edge_newline() = assertSingle("edge_cases/newline")
    @Test fun edge_crlf() = assertSingle("edge_cases/crlf")
    @Test fun edge_mixedWhitespace() = assertSingle("edge_cases/mixed_whitespace")
    @Test fun edge_leadingSpaces() = assertSingle("edge_cases/leading_spaces")
    @Test fun edge_trailingSpaces() = assertSingle("edge_cases/trailing_spaces")
    @Test fun edge_onlyNewlines() = assertSingle("edge_cases/only_newlines")
    @Test fun edge_nullChar() = assertSingle("edge_cases/null_char")
    @Test fun edge_nbsp() = assertSingle("edge_cases/nbsp")
    @Test fun edge_zeroWidthSpace() = assertSingle("edge_cases/zero_width_space")
    @Test fun edge_bom() = assertSingle("edge_cases/bom")
    @Test fun edge_veryLongWord() = assertSingle("edge_cases/very_long_word")
    @Test fun edge_veryLongNumber() = assertSingle("edge_cases/very_long_number")
    @Test fun edge_repeatedPunct() = assertSingle("edge_cases/repeated_punct")
    @Test fun edge_singleEmoji() = assertSingle("edge_cases/single_emoji")
    @Test fun edge_onlySpecial() = assertSingle("edge_cases/only_special")

    // ═══════════════════════════════════════════════════════════
    // 5. 혼합 언어 (11 TC)
    // ═══════════════════════════════════════════════════════════

    @Test fun mixed_koEn() = assertSingle("mixed_lang/ko_en")
    @Test fun mixed_jaEn() = assertSingle("mixed_lang/ja_en")
    @Test fun mixed_zhEn() = assertSingle("mixed_lang/zh_en")
    @Test fun mixed_koJa() = assertSingle("mixed_lang/ko_ja")
    @Test fun mixed_arEn() = assertSingle("mixed_lang/ar_en")
    @Test fun mixed_ruEn() = assertSingle("mixed_lang/ru_en")
    @Test fun mixed_thEn() = assertSingle("mixed_lang/th_en")
    @Test fun mixed_hiEn() = assertSingle("mixed_lang/hi_en")
    @Test fun mixed_3lang() = assertSingle("mixed_lang/multi_3lang")
    @Test fun mixed_4lang() = assertSingle("mixed_lang/multi_4lang")
    @Test fun mixed_codeMixedKo() = assertSingle("mixed_lang/code_mixed_ko")

    // ═══════════════════════════════════════════════════════════
    // 6. 숫자 / URL / 코드 (13 TC)
    // ═══════════════════════════════════════════════════════════

    @Test fun code_integers() = assertSingle("numbers_code/integers")
    @Test fun code_floats() = assertSingle("numbers_code/floats")
    @Test fun code_negative() = assertSingle("numbers_code/negative")
    @Test fun code_phone() = assertSingle("numbers_code/phone")
    @Test fun code_date() = assertSingle("numbers_code/date")
    @Test fun code_url() = assertSingle("numbers_code/url")
    @Test fun code_email() = assertSingle("numbers_code/email")
    @Test fun code_path() = assertSingle("numbers_code/path")
    @Test fun code_python() = assertSingle("numbers_code/code_python")
    @Test fun code_json() = assertSingle("numbers_code/code_json")
    @Test fun code_ipAddress() = assertSingle("numbers_code/ip_address")
    @Test fun code_hex() = assertSingle("numbers_code/hex")
    @Test fun code_version() = assertSingle("numbers_code/version")

    // ═══════════════════════════════════════════════════════════
    // 7. 반복 / 유니코드 경계 (12 TC)
    // ═══════════════════════════════════════════════════════════

    @Test fun stress_repeatedWord() = assertSingle("stress/repeated_word")
    @Test fun stress_alternating() = assertSingle("stress/alternating")
    @Test fun stress_unicodeBoundary() = assertSingle("stress/unicode_boundary")
    @Test fun stress_hangulJamo() = assertSingle("stress/hangul_jamo")
    @Test fun stress_hangulCompat() = assertSingle("stress/hangul_compat")
    @Test fun stress_surrogateEmoji() = assertSingle("stress/surrogate_emoji")
    @Test fun stress_rareCjk() = assertSingle("stress/rare_cjk")
    @Test fun stress_rtlBidi() = assertSingle("stress/rtl_bidi")
    @Test fun stress_zalgo() = assertSingle("stress/zalgo")
    @Test fun stress_ligatures() = assertSingle("stress/ligatures")
    @Test fun stress_accentedChars() = assertSingle("stress/accented_chars")
    @Test fun stress_turkishI() = assertSingle("stress/turkish_i")

    // ═══════════════════════════════════════════════════════════
    // 카테고리 전체 테스트 (한 번에 돌리기)
    // ═══════════════════════════════════════════════════════════

    @Test fun allLangBasic() = assertEncode("lang_basic")
    @Test fun allLangLong() = assertEncode("lang_long")
    @Test fun allSpecialChars() = assertEncode("special_chars")
    @Test fun allEdgeCases() = assertEncode("edge_cases")
    @Test fun allMixedLang() = assertEncode("mixed_lang")
    @Test fun allNumbersCode() = assertEncode("numbers_code")
    @Test fun allStress() = assertEncode("stress")

    // ═══════════════════════════════════════════════════════════
    // Determinism: 같은 입력을 두 번 encode해도 같은 결과
    // ═══════════════════════════════════════════════════════════

    @Test
    fun determinism() {
        val texts = listOf(
            "Hello, nice to meet you",
            "안녕하세요, 반갑습니다",
            "Hello 😀 World 🌍",
            "def encode(self, text: str) -> List[int]:",
        )
        for (text in texts) {
            val first = tokenizer.encode(text).inputIds.toList()
            val second = tokenizer.encode(text).inputIds.toList()
            assertEquals("Determinism failed for: ${text.take(30)}", first, second)
        }
    }

    // ═══════════════════════════════════════════════════════════
    // EOS 토큰이 항상 마지막에 있는지
    // ═══════════════════════════════════════════════════════════

    @Test
    fun eosTokenPosition() {
        val texts = listOf("Hello", "안녕하세요", "Test 123", "")
        val eosId = 41775
        for (text in texts) {
            val ids = tokenizer.encode(text).inputIds
            if (ids.isNotEmpty()) {
                assertEquals(
                    "EOS should be last token for: ${text.take(20)}",
                    eosId, ids.last()
                )
            }
        }
    }

    // ═══════════════════════════════════════════════════════════
    // Round-trip: encode → decode ≈ original
    // ═══════════════════════════════════════════════════════════

    @Test
    fun roundTrip() {
        val texts = listOf(
            "Hello, nice to meet you",
            "The quick brown fox jumps over the lazy dog",
        )
        for (text in texts) {
            val encoded = tokenizer.encode(text)
            val decoded = tokenizer.decode(encoded.inputIds)
            assertEquals("Round-trip failed", text, decoded)
        }
    }

    // ═══════════════════════════════════════════════════════════
    // AttentionMask 길이가 inputIds와 동일한지
    // ═══════════════════════════════════════════════════════════

    @Test
    fun attentionMaskLength() {
        val texts = listOf("Hello world", "안녕하세요", "a b c d e")
        for (text in texts) {
            val encoded = tokenizer.encode(text)
            assertEquals(
                "attentionMask length mismatch",
                encoded.inputIds.size, encoded.attentionMask.size
            )
            assertTrue(
                "attentionMask should be all 1s",
                encoded.attentionMask.all { it == 1 }
            )
        }
    }

    // ═══════════════════════════════════════════════════════════
    // 개별 TC assert (golden 비교)
    // ═══════════════════════════════════════════════════════════

    private fun assertSingle(key: String) {
        val (text, expectedIds) = golden[key]
            ?: fail("Golden key not found: $key").let { return }
        val actual = tokenizer.encode(text).inputIds.toList()
        assertEquals(
            "BPE mismatch [$key]: text=${text.take(40)}",
            expectedIds, actual
        )
    }
}
