// Language and provider compatibility data
// Based on official documentation:
// - Deepgram Nova-3: 31 languages
// - Google Cloud Speech-to-Text v2: 85+ languages  
// - OpenAI Realtime: ~50 languages
// - DeepL: 33 languages
// - Google Cloud Translation: 189 languages
// - OpenAI GPT: 100+ languages

export interface LanguageInfo {
  code: string;
  name: string;
  nativeName: string;
  // Which STT providers support this language
  sttProviders: ('deepgram' | 'openai' | 'google')[];
  // Which translation providers support this language
  translationProviders: ('deepl' | 'openai' | 'google')[];
}

// Deepgram Nova-3 supported languages (31 languages)
const DEEPGRAM_LANGUAGES = new Set([
  'zh', 'zh-CN', 'zh-TW', // Chinese
  'da', // Danish
  'nl', // Dutch
  'en', 'en-AU', 'en-GB', 'en-IN', 'en-NZ', 'en-US', // English
  'nl-BE', // Flemish
  'fr', 'fr-CA', // French
  'de', // German
  'de-CH', // German (Swiss)
  'hi', // Hindi
  'hi-Latn', // Hindi (Latin)
  'id', // Indonesian
  'it', // Italian
  'ja', // Japanese
  'ko', // Korean
  'no', // Norwegian
  'pl', // Polish
  'pt', 'pt-BR', 'pt-PT', // Portuguese
  'ru', // Russian
  'es', 'es-419', // Spanish
  'sv', // Swedish
  'ta', // Tamil
  'tr', // Turkish
  'uk', // Ukrainian
]);

// OpenAI Realtime/Whisper supported languages (~50 languages)
const OPENAI_STT_LANGUAGES = new Set([
  'af', // Afrikaans
  'ar', // Arabic
  'hy', // Armenian
  'az', // Azerbaijani
  'be', // Belarusian
  'bs', // Bosnian
  'bg', // Bulgarian
  'ca', // Catalan
  'zh', // Chinese
  'hr', // Croatian
  'cs', // Czech
  'da', // Danish
  'nl', // Dutch
  'en', // English
  'et', // Estonian
  'fi', // Finnish
  'fr', // French
  'gl', // Galician
  'de', // German
  'el', // Greek
  'he', // Hebrew
  'hi', // Hindi
  'hu', // Hungarian
  'is', // Icelandic
  'id', // Indonesian
  'it', // Italian
  'ja', // Japanese
  'kk', // Kazakh
  'ko', // Korean
  'lv', // Latvian
  'lt', // Lithuanian
  'mk', // Macedonian
  'ms', // Malay
  'mr', // Marathi
  'mi', // Maori
  'ne', // Nepali
  'no', // Norwegian
  'fa', // Persian
  'pl', // Polish
  'pt', // Portuguese
  'ro', // Romanian
  'ru', // Russian
  'sr', // Serbian
  'sk', // Slovak
  'sl', // Slovenian
  'es', // Spanish
  'sw', // Swahili
  'sv', // Swedish
  'tl', // Tagalog
  'ta', // Tamil
  'th', // Thai
  'tr', // Turkish
  'uk', // Ukrainian
  'ur', // Urdu
  'vi', // Vietnamese
  'cy', // Welsh
]);

// Google Cloud Speech-to-Text v2 supported languages (85+ languages)
const GOOGLE_STT_LANGUAGES = new Set([
  'af', 'am', 'ar', 'az', 'be', 'bg', 'bn', 'bs', 'ca', 'ceb', 'co', 
  'cs', 'cy', 'da', 'de', 'el', 'en', 'eo', 'es', 'et', 'eu', 'fa', 
  'fi', 'fil', 'fr', 'fy', 'ga', 'gd', 'gl', 'gu', 'ha', 'haw', 'he', 
  'hi', 'hmn', 'hr', 'ht', 'hu', 'hy', 'id', 'ig', 'is', 'it', 'iw', 
  'ja', 'jw', 'ka', 'kk', 'km', 'kn', 'ko', 'ku', 'ky', 'la', 'lb', 
  'lo', 'lt', 'lv', 'mg', 'mi', 'mk', 'ml', 'mn', 'mr', 'ms', 'mt', 
  'my', 'ne', 'nl', 'no', 'ny', 'or', 'pa', 'pl', 'ps', 'pt', 'ro', 
  'ru', 'rw', 'sd', 'si', 'sk', 'sl', 'sm', 'sn', 'so', 'sq', 'sr', 
  'st', 'su', 'sv', 'sw', 'ta', 'te', 'tg', 'th', 'tk', 'tl', 'tr', 
  'tt', 'ug', 'uk', 'ur', 'uz', 'vi', 'xh', 'yi', 'yo', 'zh', 'zu'
]);

// DeepL supported languages (33 languages)
const DEEPL_LANGUAGES = new Set([
  'ar', // Arabic
  'bg', // Bulgarian
  'cs', // Czech
  'da', // Danish
  'de', // German
  'el', // Greek
  'en', // English
  'es', // Spanish
  'et', // Estonian
  'fi', // Finnish
  'fr', // French
  'hu', // Hungarian
  'id', // Indonesian
  'it', // Italian
  'ja', // Japanese
  'ko', // Korean
  'lt', // Lithuanian
  'lv', // Latvian
  'nb', // Norwegian (Bokmål)
  'nl', // Dutch
  'pl', // Polish
  'pt', // Portuguese
  'ro', // Romanian
  'ru', // Russian
  'sk', // Slovak
  'sl', // Slovenian
  'sv', // Swedish
  'tr', // Turkish
  'uk', // Ukrainian
  'zh', // Chinese
]);

// Google Cloud Translation supported languages (189 languages - we include most common)
const GOOGLE_TRANSLATE_LANGUAGES = new Set([
  'af', 'am', 'ar', 'az', 'be', 'bg', 'bn', 'bs', 'ca', 'ceb', 'co', 
  'cs', 'cy', 'da', 'de', 'el', 'en', 'eo', 'es', 'et', 'eu', 'fa', 
  'fi', 'fil', 'fr', 'fy', 'ga', 'gd', 'gl', 'gu', 'ha', 'haw', 'he', 
  'hi', 'hmn', 'hr', 'ht', 'hu', 'hy', 'id', 'ig', 'is', 'it', 'iw', 
  'ja', 'jw', 'ka', 'kk', 'km', 'kn', 'ko', 'ku', 'ky', 'la', 'lb', 
  'lo', 'lt', 'lv', 'mg', 'mi', 'mk', 'ml', 'mn', 'mr', 'ms', 'mt', 
  'my', 'ne', 'nl', 'no', 'ny', 'or', 'pa', 'pl', 'ps', 'pt', 'ro', 
  'ru', 'rw', 'sd', 'si', 'sk', 'sl', 'sm', 'sn', 'so', 'sq', 'sr', 
  'st', 'su', 'sv', 'sw', 'ta', 'te', 'tg', 'th', 'tk', 'tl', 'tr', 
  'tt', 'ug', 'uk', 'ur', 'uz', 'vi', 'xh', 'yi', 'yo', 'zh', 'zu'
]);

// OpenAI GPT translation (supports virtually all languages)
const OPENAI_TRANSLATE_LANGUAGES = new Set([
  ...GOOGLE_TRANSLATE_LANGUAGES, // All Google languages plus more
]);

// Complete language list with provider compatibility
export const LANGUAGES: LanguageInfo[] = [
  { code: 'auto', name: 'Auto detect', nativeName: 'Auto', sttProviders: ['deepgram', 'openai', 'google'], translationProviders: [] },
  { code: 'af', name: 'Afrikaans', nativeName: 'Afrikaans', sttProviders: ['openai', 'google'], translationProviders: ['google', 'openai'] },
  { code: 'am', name: 'Amharic', nativeName: 'አማርኛ', sttProviders: ['google'], translationProviders: ['google', 'openai'] },
  { code: 'ar', name: 'Arabic', nativeName: 'العربية', sttProviders: ['openai', 'google'], translationProviders: ['deepl', 'google', 'openai'] },
  { code: 'az', name: 'Azerbaijani', nativeName: 'Azərbaycan', sttProviders: ['openai', 'google'], translationProviders: ['google', 'openai'] },
  { code: 'be', name: 'Belarusian', nativeName: 'Беларуская', sttProviders: ['openai', 'google'], translationProviders: ['google', 'openai'] },
  { code: 'bg', name: 'Bulgarian', nativeName: 'Български', sttProviders: ['openai', 'google'], translationProviders: ['deepl', 'google', 'openai'] },
  { code: 'bn', name: 'Bengali', nativeName: 'বাংলা', sttProviders: ['google'], translationProviders: ['google', 'openai'] },
  { code: 'bs', name: 'Bosnian', nativeName: 'Bosanski', sttProviders: ['openai', 'google'], translationProviders: ['google', 'openai'] },
  { code: 'ca', name: 'Catalan', nativeName: 'Català', sttProviders: ['openai', 'google'], translationProviders: ['google', 'openai'] },
  { code: 'ceb', name: 'Cebuano', nativeName: 'Cebuano', sttProviders: ['google'], translationProviders: ['google', 'openai'] },
  { code: 'co', name: 'Corsican', nativeName: 'Corsu', sttProviders: ['google'], translationProviders: ['google', 'openai'] },
  { code: 'cs', name: 'Czech', nativeName: 'Čeština', sttProviders: ['openai', 'google'], translationProviders: ['deepl', 'google', 'openai'] },
  { code: 'cy', name: 'Welsh', nativeName: 'Cymraeg', sttProviders: ['openai', 'google'], translationProviders: ['google', 'openai'] },
  { code: 'da', name: 'Danish', nativeName: 'Dansk', sttProviders: ['deepgram', 'openai', 'google'], translationProviders: ['deepl', 'google', 'openai'] },
  { code: 'de', name: 'German', nativeName: 'Deutsch', sttProviders: ['deepgram', 'openai', 'google'], translationProviders: ['deepl', 'google', 'openai'] },
  { code: 'el', name: 'Greek', nativeName: 'Ελληνικά', sttProviders: ['openai', 'google'], translationProviders: ['deepl', 'google', 'openai'] },
  { code: 'en', name: 'English', nativeName: 'English', sttProviders: ['deepgram', 'openai', 'google'], translationProviders: ['deepl', 'google', 'openai'] },
  { code: 'eo', name: 'Esperanto', nativeName: 'Esperanto', sttProviders: ['google'], translationProviders: ['google', 'openai'] },
  { code: 'es', name: 'Spanish', nativeName: 'Español', sttProviders: ['deepgram', 'openai', 'google'], translationProviders: ['deepl', 'google', 'openai'] },
  { code: 'et', name: 'Estonian', nativeName: 'Eesti', sttProviders: ['openai', 'google'], translationProviders: ['deepl', 'google', 'openai'] },
  { code: 'eu', name: 'Basque', nativeName: 'Euskara', sttProviders: ['google'], translationProviders: ['google', 'openai'] },
  { code: 'fa', name: 'Persian', nativeName: 'فارسی', sttProviders: ['openai', 'google'], translationProviders: ['google', 'openai'] },
  { code: 'fi', name: 'Finnish', nativeName: 'Suomi', sttProviders: ['openai', 'google'], translationProviders: ['deepl', 'google', 'openai'] },
  { code: 'fil', name: 'Filipino', nativeName: 'Filipino', sttProviders: ['google'], translationProviders: ['google', 'openai'] },
  { code: 'fr', name: 'French', nativeName: 'Français', sttProviders: ['deepgram', 'openai', 'google'], translationProviders: ['deepl', 'google', 'openai'] },
  { code: 'fy', name: 'Frisian', nativeName: 'Frysk', sttProviders: ['google'], translationProviders: ['google', 'openai'] },
  { code: 'ga', name: 'Irish', nativeName: 'Gaeilge', sttProviders: ['google'], translationProviders: ['google', 'openai'] },
  { code: 'gd', name: 'Scottish Gaelic', nativeName: 'Gàidhlig', sttProviders: ['google'], translationProviders: ['google', 'openai'] },
  { code: 'gl', name: 'Galician', nativeName: 'Galego', sttProviders: ['openai', 'google'], translationProviders: ['google', 'openai'] },
  { code: 'gu', name: 'Gujarati', nativeName: 'ગુજરાતી', sttProviders: ['google'], translationProviders: ['google', 'openai'] },
  { code: 'ha', name: 'Hausa', nativeName: 'Hausa', sttProviders: ['google'], translationProviders: ['google', 'openai'] },
  { code: 'haw', name: 'Hawaiian', nativeName: 'ʻŌlelo Hawaiʻi', sttProviders: ['google'], translationProviders: ['google', 'openai'] },
  { code: 'he', name: 'Hebrew', nativeName: 'עברית', sttProviders: ['openai', 'google'], translationProviders: ['google', 'openai'] },
  { code: 'hi', name: 'Hindi', nativeName: 'हिन्दी', sttProviders: ['deepgram', 'openai', 'google'], translationProviders: ['google', 'openai'] },
  { code: 'hmn', name: 'Hmong', nativeName: 'Hmong', sttProviders: ['google'], translationProviders: ['google', 'openai'] },
  { code: 'hr', name: 'Croatian', nativeName: 'Hrvatski', sttProviders: ['openai', 'google'], translationProviders: ['google', 'openai'] },
  { code: 'ht', name: 'Haitian Creole', nativeName: 'Kreyòl Ayisyen', sttProviders: ['google'], translationProviders: ['google', 'openai'] },
  { code: 'hu', name: 'Hungarian', nativeName: 'Magyar', sttProviders: ['openai', 'google'], translationProviders: ['deepl', 'google', 'openai'] },
  { code: 'hy', name: 'Armenian', nativeName: 'Հայերdelays', sttProviders: ['openai', 'google'], translationProviders: ['google', 'openai'] },
  { code: 'id', name: 'Indonesian', nativeName: 'Bahasa Indonesia', sttProviders: ['deepgram', 'openai', 'google'], translationProviders: ['deepl', 'google', 'openai'] },
  { code: 'ig', name: 'Igbo', nativeName: 'Igbo', sttProviders: ['google'], translationProviders: ['google', 'openai'] },
  { code: 'is', name: 'Icelandic', nativeName: 'Íslenska', sttProviders: ['openai', 'google'], translationProviders: ['google', 'openai'] },
  { code: 'it', name: 'Italian', nativeName: 'Italiano', sttProviders: ['deepgram', 'openai', 'google'], translationProviders: ['deepl', 'google', 'openai'] },
  { code: 'ja', name: 'Japanese', nativeName: '日本語', sttProviders: ['deepgram', 'openai', 'google'], translationProviders: ['deepl', 'google', 'openai'] },
  { code: 'jw', name: 'Javanese', nativeName: 'Basa Jawa', sttProviders: ['google'], translationProviders: ['google', 'openai'] },
  { code: 'ka', name: 'Georgian', nativeName: 'ქართული', sttProviders: ['google'], translationProviders: ['google', 'openai'] },
  { code: 'kk', name: 'Kazakh', nativeName: 'Қазақ', sttProviders: ['openai', 'google'], translationProviders: ['google', 'openai'] },
  { code: 'km', name: 'Khmer', nativeName: 'ខ្មែរ', sttProviders: ['google'], translationProviders: ['google', 'openai'] },
  { code: 'kn', name: 'Kannada', nativeName: 'ಕನ್ನಡ', sttProviders: ['google'], translationProviders: ['google', 'openai'] },
  { code: 'ko', name: 'Korean', nativeName: '한국어', sttProviders: ['deepgram', 'openai', 'google'], translationProviders: ['deepl', 'google', 'openai'] },
  { code: 'ku', name: 'Kurdish', nativeName: 'Kurdî', sttProviders: ['google'], translationProviders: ['google', 'openai'] },
  { code: 'ky', name: 'Kyrgyz', nativeName: 'Кыргызча', sttProviders: ['google'], translationProviders: ['google', 'openai'] },
  { code: 'la', name: 'Latin', nativeName: 'Latina', sttProviders: ['google'], translationProviders: ['google', 'openai'] },
  { code: 'lb', name: 'Luxembourgish', nativeName: 'Lëtzebuergesch', sttProviders: ['google'], translationProviders: ['google', 'openai'] },
  { code: 'lo', name: 'Lao', nativeName: 'ລາວ', sttProviders: ['google'], translationProviders: ['google', 'openai'] },
  { code: 'lt', name: 'Lithuanian', nativeName: 'Lietuvių', sttProviders: ['openai', 'google'], translationProviders: ['deepl', 'google', 'openai'] },
  { code: 'lv', name: 'Latvian', nativeName: 'Latviešu', sttProviders: ['openai', 'google'], translationProviders: ['deepl', 'google', 'openai'] },
  { code: 'mg', name: 'Malagasy', nativeName: 'Malagasy', sttProviders: ['google'], translationProviders: ['google', 'openai'] },
  { code: 'mi', name: 'Maori', nativeName: 'Māori', sttProviders: ['openai', 'google'], translationProviders: ['google', 'openai'] },
  { code: 'mk', name: 'Macedonian', nativeName: 'Македонски', sttProviders: ['openai', 'google'], translationProviders: ['google', 'openai'] },
  { code: 'ml', name: 'Malayalam', nativeName: 'മലയാളം', sttProviders: ['google'], translationProviders: ['google', 'openai'] },
  { code: 'mn', name: 'Mongolian', nativeName: 'Монгол', sttProviders: ['google'], translationProviders: ['google', 'openai'] },
  { code: 'mr', name: 'Marathi', nativeName: 'मराठी', sttProviders: ['openai', 'google'], translationProviders: ['google', 'openai'] },
  { code: 'ms', name: 'Malay', nativeName: 'Bahasa Melayu', sttProviders: ['openai', 'google'], translationProviders: ['google', 'openai'] },
  { code: 'mt', name: 'Maltese', nativeName: 'Malti', sttProviders: ['google'], translationProviders: ['google', 'openai'] },
  { code: 'my', name: 'Myanmar (Burmese)', nativeName: 'မြန်မာ', sttProviders: ['google'], translationProviders: ['google', 'openai'] },
  { code: 'ne', name: 'Nepali', nativeName: 'नेपाली', sttProviders: ['openai', 'google'], translationProviders: ['google', 'openai'] },
  { code: 'nl', name: 'Dutch', nativeName: 'Nederlands', sttProviders: ['deepgram', 'openai', 'google'], translationProviders: ['deepl', 'google', 'openai'] },
  { code: 'no', name: 'Norwegian', nativeName: 'Norsk', sttProviders: ['deepgram', 'openai', 'google'], translationProviders: ['deepl', 'google', 'openai'] },
  { code: 'ny', name: 'Chichewa', nativeName: 'Chichewa', sttProviders: ['google'], translationProviders: ['google', 'openai'] },
  { code: 'or', name: 'Odia', nativeName: 'ଓଡ଼ିଆ', sttProviders: ['google'], translationProviders: ['google', 'openai'] },
  { code: 'pa', name: 'Punjabi', nativeName: 'ਪੰਜਾਬੀ', sttProviders: ['google'], translationProviders: ['google', 'openai'] },
  { code: 'pl', name: 'Polish', nativeName: 'Polski', sttProviders: ['deepgram', 'openai', 'google'], translationProviders: ['deepl', 'google', 'openai'] },
  { code: 'ps', name: 'Pashto', nativeName: 'پښتو', sttProviders: ['google'], translationProviders: ['google', 'openai'] },
  { code: 'pt', name: 'Portuguese', nativeName: 'Português', sttProviders: ['deepgram', 'openai', 'google'], translationProviders: ['deepl', 'google', 'openai'] },
  { code: 'ro', name: 'Romanian', nativeName: 'Română', sttProviders: ['openai', 'google'], translationProviders: ['deepl', 'google', 'openai'] },
  { code: 'ru', name: 'Russian', nativeName: 'Русский', sttProviders: ['deepgram', 'openai', 'google'], translationProviders: ['deepl', 'google', 'openai'] },
  { code: 'rw', name: 'Kinyarwanda', nativeName: 'Kinyarwanda', sttProviders: ['google'], translationProviders: ['google', 'openai'] },
  { code: 'sd', name: 'Sindhi', nativeName: 'سنڌي', sttProviders: ['google'], translationProviders: ['google', 'openai'] },
  { code: 'si', name: 'Sinhala', nativeName: 'සිංහල', sttProviders: ['google'], translationProviders: ['google', 'openai'] },
  { code: 'sk', name: 'Slovak', nativeName: 'Slovenčina', sttProviders: ['openai', 'google'], translationProviders: ['deepl', 'google', 'openai'] },
  { code: 'sl', name: 'Slovenian', nativeName: 'Slovenščina', sttProviders: ['openai', 'google'], translationProviders: ['deepl', 'google', 'openai'] },
  { code: 'sm', name: 'Samoan', nativeName: 'Gagana Samoa', sttProviders: ['google'], translationProviders: ['google', 'openai'] },
  { code: 'sn', name: 'Shona', nativeName: 'Shona', sttProviders: ['google'], translationProviders: ['google', 'openai'] },
  { code: 'so', name: 'Somali', nativeName: 'Soomaali', sttProviders: ['google'], translationProviders: ['google', 'openai'] },
  { code: 'sq', name: 'Albanian', nativeName: 'Shqip', sttProviders: ['google'], translationProviders: ['google', 'openai'] },
  { code: 'sr', name: 'Serbian', nativeName: 'Српски', sttProviders: ['openai', 'google'], translationProviders: ['google', 'openai'] },
  { code: 'st', name: 'Sesotho', nativeName: 'Sesotho', sttProviders: ['google'], translationProviders: ['google', 'openai'] },
  { code: 'su', name: 'Sundanese', nativeName: 'Basa Sunda', sttProviders: ['google'], translationProviders: ['google', 'openai'] },
  { code: 'sv', name: 'Swedish', nativeName: 'Svenska', sttProviders: ['deepgram', 'openai', 'google'], translationProviders: ['deepl', 'google', 'openai'] },
  { code: 'sw', name: 'Swahili', nativeName: 'Kiswahili', sttProviders: ['openai', 'google'], translationProviders: ['google', 'openai'] },
  { code: 'ta', name: 'Tamil', nativeName: 'தமிழ்', sttProviders: ['deepgram', 'openai', 'google'], translationProviders: ['google', 'openai'] },
  { code: 'te', name: 'Telugu', nativeName: 'తెలుగు', sttProviders: ['google'], translationProviders: ['google', 'openai'] },
  { code: 'tg', name: 'Tajik', nativeName: 'Тоҷикӣ', sttProviders: ['google'], translationProviders: ['google', 'openai'] },
  { code: 'th', name: 'Thai', nativeName: 'ไทย', sttProviders: ['openai', 'google'], translationProviders: ['google', 'openai'] },
  { code: 'tk', name: 'Turkmen', nativeName: 'Türkmen', sttProviders: ['google'], translationProviders: ['google', 'openai'] },
  { code: 'tl', name: 'Tagalog', nativeName: 'Tagalog', sttProviders: ['openai', 'google'], translationProviders: ['google', 'openai'] },
  { code: 'tr', name: 'Turkish', nativeName: 'Türkçe', sttProviders: ['deepgram', 'openai', 'google'], translationProviders: ['deepl', 'google', 'openai'] },
  { code: 'tt', name: 'Tatar', nativeName: 'Татар', sttProviders: ['google'], translationProviders: ['google', 'openai'] },
  { code: 'ug', name: 'Uyghur', nativeName: 'ئۇيغۇرچە', sttProviders: ['google'], translationProviders: ['google', 'openai'] },
  { code: 'uk', name: 'Ukrainian', nativeName: 'Українська', sttProviders: ['deepgram', 'openai', 'google'], translationProviders: ['deepl', 'google', 'openai'] },
  { code: 'ur', name: 'Urdu', nativeName: 'اردو', sttProviders: ['openai', 'google'], translationProviders: ['google', 'openai'] },
  { code: 'uz', name: 'Uzbek', nativeName: 'Oʻzbek', sttProviders: ['google'], translationProviders: ['google', 'openai'] },
  { code: 'vi', name: 'Vietnamese', nativeName: 'Tiếng Việt', sttProviders: ['openai', 'google'], translationProviders: ['google', 'openai'] },
  { code: 'xh', name: 'Xhosa', nativeName: 'isiXhosa', sttProviders: ['google'], translationProviders: ['google', 'openai'] },
  { code: 'yi', name: 'Yiddish', nativeName: 'ייִדיש', sttProviders: ['google'], translationProviders: ['google', 'openai'] },
  { code: 'yo', name: 'Yoruba', nativeName: 'Yorùbá', sttProviders: ['google'], translationProviders: ['google', 'openai'] },
  { code: 'zh', name: 'Chinese', nativeName: '中文', sttProviders: ['deepgram', 'openai', 'google'], translationProviders: ['deepl', 'google', 'openai'] },
  { code: 'zu', name: 'Zulu', nativeName: 'isiZulu', sttProviders: ['google'], translationProviders: ['google', 'openai'] },
];

// Provider info with supported language count
export const STT_PROVIDERS = {
  deepgram: {
    id: 'deepgram',
    name: 'Deepgram Nova-3',
    description: 'Fastest realtime (~300ms)',
    languageCount: 31,
    icon: '⚡',
  },
  openai: {
    id: 'openai',
    name: 'OpenAI Realtime',
    description: 'Quality mode (~1-2s)',
    languageCount: 50,
    icon: '🤖',
  },
  google: {
    id: 'google',
    name: 'Google Cloud STT v2',
    description: 'Most languages (~500ms)',
    languageCount: 85,
    icon: '🔵',
  },
};

export const TRANSLATION_PROVIDERS = {
  deepl: {
    id: 'deepl',
    name: 'DeepL',
    description: 'Best EU quality, low-latency',
    languageCount: 33,
    icon: '💎',
  },
  google: {
    id: 'google',
    name: 'Google Cloud',
    description: 'Widest coverage (189 languages)',
    languageCount: 189,
    icon: '🌍',
  },
  openai: {
    id: 'openai',
    name: 'OpenAI GPT-4',
    description: 'Quality translation',
    languageCount: 100,
    icon: '🤖',
  },
};

// Helper functions
export function getLanguageByCode(code: string): LanguageInfo | undefined {
  return LANGUAGES.find(lang => lang.code === code);
}

export function getLanguagesForSTTProvider(provider: 'deepgram' | 'openai' | 'google'): LanguageInfo[] {
  return LANGUAGES.filter(lang => lang.sttProviders.includes(provider));
}

export function getLanguagesForTranslationProvider(provider: 'deepl' | 'openai' | 'google'): LanguageInfo[] {
  return LANGUAGES.filter(lang => lang.translationProviders.includes(provider));
}

export function getSTTProvidersForLanguage(langCode: string): ('deepgram' | 'openai' | 'google')[] {
  if (langCode === 'auto') return ['deepgram', 'openai', 'google'];
  const lang = getLanguageByCode(langCode);
  return lang?.sttProviders || [];
}

export function getTranslationProvidersForLanguage(langCode: string): ('deepl' | 'openai' | 'google')[] {
  const lang = getLanguageByCode(langCode);
  return lang?.translationProviders || ['openai']; // OpenAI fallback for unknown languages
}

export function searchLanguages(query: string): LanguageInfo[] {
  const q = query.toLowerCase().trim();
  if (!q) return LANGUAGES;
  
  return LANGUAGES.filter(lang => 
    lang.name.toLowerCase().includes(q) ||
    lang.nativeName.toLowerCase().includes(q) ||
    lang.code.toLowerCase().includes(q)
  );
}
