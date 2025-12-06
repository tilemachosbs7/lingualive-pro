/**
 * =============================================================================
 * I18N/MESSAGES.TS — Internationalization Layer for LinguaLive Pro Extension
 * =============================================================================
 *
 * Purpose:
 * --------
 * Provides a simple but solid i18n foundation for the extension UI.
 * Centralizes all user-facing strings with support for Greek (primary) and English.
 *
 * Usage:
 * ------
 * import { t } from '../i18n/messages';
 * const text = t('hud.status.connecting');
 *
 * Notes:
 * ------
 * - Greek ('el') is the primary/default language.
 * - Locale is auto-detected from navigator.language, falling back to 'el'.
 * - No runtime locale settings yet; this is a foundation layer.
 * =============================================================================
 */

// =============================================================================
// TYPES
// =============================================================================

/**
 * Supported locales.
 * 'el' = Greek (primary), 'en' = English
 */
export type Locale = 'el' | 'en';

/**
 * Message keys for type-safe i18n.
 * Using a string union so TypeScript can validate key usage.
 */
export type MessageKey =
  // HUD: Title & Description
  | 'hud.title'
  | 'hud.description'

  // HUD: Status strings
  | 'hud.status.idle'
  | 'hud.status.connecting'
  | 'hud.status.connected'
  | 'hud.status.sessionStarted'
  | 'hud.status.live'
  | 'hud.status.stopping'
  | 'hud.status.disconnected'
  | 'hud.status.error'
  | 'hud.status.noAuth'
  | 'hud.status.authError'
  | 'hud.status.usageLimit'
  | 'hud.status.planLimit'
  | 'hud.status.usingFastMode'
  | 'hud.status.loadingLimits'
  | 'hud.status.loadingUsage'

  // HUD: Auth required
  | 'hud.auth.required'
  | 'hud.auth.tokenExpired'
  | 'hud.auth.openSettings'

  // HUD: Limits exceeded
  | 'hud.limits.reached'
  | 'hud.limits.sessionReached'
  | 'hud.limits.rateLimited'
  | 'hud.limits.upgrade'
  | 'hud.limits.startButtonTitle'

  // HUD: Diagnostics link
  | 'hud.diagnostics.link'

  // HUD: Smart mode warning
  | 'hud.smartWarning.notAvailable'
  | 'hud.smartWarning.upgradeHint'

  // HUD: Transcript placeholder
  | 'hud.transcript.placeholder'

  // HUD: Upgrade nudge
  | 'hud.upgrade.link'

  // HUD: Usage display
  | 'hud.usage.label'
  | 'hud.usage.minutes'
  | 'hud.usage.sessions'
  | 'hud.usage.sessionSingular'

  // Options: Diagnostics section
  | 'options.diagnostics.title'
  | 'options.diagnostics.description'
  | 'options.diagnostics.runButton'
  | 'options.diagnostics.runningButton'
  | 'options.diagnostics.copyButton'
  | 'options.diagnostics.copyFirst'
  | 'options.diagnostics.copySuccess'
  | 'options.diagnostics.copyFailed'
  | 'options.diagnostics.clipboardNotSupported'
  | 'options.diagnostics.checkError'
  | 'options.diagnostics.reportTitle'
  | 'options.diagnostics.results'

  // Options: Diagnostic result names
  | 'options.diagnostics.result.environment'
  | 'options.diagnostics.result.backend'
  | 'options.diagnostics.result.auth'
  | 'options.diagnostics.result.microphone'

  // Options: Diagnostic messages
  | 'options.diagnostics.env.ok'
  | 'options.diagnostics.env.chromeApiMissing'
  | 'options.diagnostics.env.missingFeatures'
  | 'options.diagnostics.backend.ok'
  | 'options.diagnostics.backend.unreachable'
  | 'options.diagnostics.backend.reachableAuthSeparate'
  | 'options.diagnostics.backend.skipped'
  | 'options.diagnostics.auth.notConnected'
  | 'options.diagnostics.auth.connected'
  | 'options.diagnostics.auth.invalid'
  | 'options.diagnostics.auth.unreachable'
  | 'options.diagnostics.mic.granted'
  | 'options.diagnostics.mic.denied'
  | 'options.diagnostics.mic.prompt'
  | 'options.diagnostics.mic.skipped'
  | 'options.diagnostics.mic.skippedBrowser'

  // Options: Backend status diagnostics
  | 'options.diagnostics.result.backendStatus'
  | 'options.diagnostics.backendStatus.ok'
  | 'options.diagnostics.backendStatus.unreachable'
  | 'options.diagnostics.backendStatus.error'

  // Options: Realtime WebSocket diagnostics
  | 'options.diagnostics.result.realtimeWebSocket'
  | 'options.diagnostics.realtimeWs.ok'
  | 'options.diagnostics.realtimeWs.skippedNoAuth'
  | 'options.diagnostics.realtimeWs.skippedNoBackend'
  | 'options.diagnostics.realtimeWs.error'

  // Options: Connection test
  | 'options.connection.testing'
  | 'options.connection.button'
  | 'options.connection.noAuth'
  | 'options.connection.invalidToken'
  | 'options.connection.serverNotFound'
  | 'options.connection.success'

  // Options: Generate dev token
  | 'options.backend.generateDevToken.button'
  | 'options.backend.generateDevToken.generating'
  | 'options.backend.generateDevToken.success'
  | 'options.backend.generateDevToken.errorMissingUser'
  | 'options.backend.generateDevToken.errorMissingBackend'
  | 'options.backend.generateDevToken.errorHttp'
  | 'options.backend.generateDevToken.errorNetwork'

  // Options: Dev token expiry display
  | 'options.backend.devTokenExpiry.label'
  | 'options.backend.devTokenExpiry.none'

  // Options: Revoke dev token
  | 'options.backend.revokeDevToken.button'
  | 'options.backend.revokeDevToken.revoking'
  | 'options.backend.revokeDevToken.success'
  | 'options.backend.revokeDevToken.errorNoToken'
  | 'options.backend.revokeDevToken.errorHttp'
  | 'options.backend.revokeDevToken.errorNetwork'

  // Options: Summary panel
  | 'options.summary.title'
  | 'options.summary.plan'
  | 'options.summary.usage'
  | 'options.summary.loading'
  | 'options.summary.loadError'
  | 'options.summary.noAuth'
  | 'options.summary.minutes'
  | 'options.summary.sessions'
  | 'options.summary.sessionSingular'
  | 'options.summary.perSession'
  | 'options.summary.maxPerSession'

  // Options: Language selector
  | 'options.language.label'
  | 'options.language.auto'
  | 'options.language.el'
  | 'options.language.en'

  // Options: Button labels
  | 'options.buttons.testConnection'
  | 'options.buttons.copyDiagnostics'

  // Options: Shortcuts & controls section
  | 'options.shortcuts.sectionTitle'
  | 'options.shortcuts.howToStart'
  | 'options.shortcuts.howToStart.steps'
  | 'options.shortcuts.keyboardHeading'
  | 'options.shortcuts.keyboardToggle'
  | 'options.shortcuts.keyboardOpenOptions'
  | 'options.shortcuts.contextMenuHeading'
  | 'options.shortcuts.contextMenuDescription'
  | 'options.shortcuts.chromeShortcutsNote'

  // Options: Help & troubleshooting section
  | 'options.help.sectionTitle'
  | 'options.help.whenToUseDiagnostics'
  | 'options.help.stepsHeading'
  | 'options.help.stepCheckBackend'
  | 'options.help.stepCheckMic'
  | 'options.help.stepCopyReport'
  | 'options.help.privacyNote'

  // Options: General
  | 'options.error.generic';

// =============================================================================
// MESSAGE TABLES
// =============================================================================

/**
 * Message table: all strings keyed by locale and message key.
 * Greek ('el') is the primary language with full coverage.
 * English ('en') provided for completeness.
 */
const MESSAGES: Record<Locale, Record<MessageKey, string>> = {
  el: {
    // HUD: Title & Description
    'hud.title': 'LinguaLive Pro',
    'hud.description': 'Real-time speech-to-text captions and translation overlay',

    // HUD: Status strings
    'hud.status.idle': 'Ανενεργό',
    'hud.status.connecting': 'Γίνεται σύνδεση…',
    'hud.status.connected': 'Συνδέθηκε',
    'hud.status.sessionStarted': 'Η συνεδρία ξεκίνησε',
    'hud.status.live': 'Ζωντανοί υπότιτλοι',
    'hud.status.stopping': 'Γίνεται διακοπή…',
    'hud.status.disconnected': 'Αποσυνδέθηκε',
    'hud.status.error': 'Σφάλμα – άνοιξε τη σελίδα ρυθμίσεων και τα διαγνωστικά για περισσότερες λεπτομέρειες.',
    'hud.status.noAuth': 'Δεν έχεις συνδεθεί',
    'hud.status.authError': 'Σφάλμα αυθεντικοποίησης',
    'hud.status.usageLimit': 'Όριο χρήσης',
    'hud.status.planLimit': 'Όριο πλάνου',
    'hud.status.usingFastMode': 'Χρήση FAST mode',
    'hud.status.loadingLimits': 'Φόρτωση ορίων…',
    'hud.status.loadingUsage': 'Χρήση: …',

    // HUD: Auth required
    'hud.auth.required': 'Για να χρησιμοποιήσεις το LinguaLive, συνδέσου πρώτα στον λογαριασμό σου.',
    'hud.auth.tokenExpired': 'Το token σου δεν είναι έγκυρο ή έχει λήξει. Άνοιξε τις ρυθμίσεις για να το ανανεώσεις.',
    'hud.auth.openSettings': 'Άνοιγμα Ρυθμίσεων',

    // HUD: Limits exceeded
    'hud.limits.reached': 'Έφτασες το όριο του πλάνου σου για σήμερα. Οι υπότιτλοι σταμάτησαν.',
    'hud.limits.sessionReached': 'Έφτασες το όριο του πλάνου σου για αυτή τη συνεδρία. Οι υπότιτλοι σταμάτησαν.',
    'hud.limits.rateLimited': 'Ξεπέρασες το όριο χρήσης. Περίμενε λίγο ή αναβάθμισε το πλάνο σου.',
    'hud.limits.upgrade': 'Αναβάθμιση',
    'hud.limits.startButtonTitle': 'Το όριο έχει ξεπεραστεί – αναβάθμισε για να συνεχίσεις',

    // HUD: Diagnostics link
    'hud.diagnostics.link': 'Διάγνωση προβλήματος',

    // HUD: Smart mode warning
    'hud.smartWarning.notAvailable': '⚠️ Το SMART mode δεν είναι διαθέσιμο.',
    'hud.smartWarning.upgradeHint': ' Αναβάθμισε για SMART mode.',

    // HUD: Transcript placeholder
    'hud.transcript.placeholder': 'Οι υπότιτλοι θα εμφανιστούν εδώ…',

    // HUD: Upgrade nudge
    'hud.upgrade.link': 'Upgrade',

    // HUD: Usage display
    'hud.usage.label': 'Χρήση',
    'hud.usage.minutes': 'λεπτά',
    'hud.usage.sessions': 'συνεδρίες',
    'hud.usage.sessionSingular': 'συνεδρία',

    // Options: Diagnostics section
    'options.diagnostics.title': 'Διαγνωστικά / Έλεγχος συστήματος',
    'options.diagnostics.description': 'Τρέξε έναν γρήγορο έλεγχο για να δεις αν όλα είναι σωστά ρυθμισμένα (backend, σύνδεση, μικρόφωνο).',
    'options.diagnostics.runButton': '🔍 Γρήγορος έλεγχος',
    'options.diagnostics.runningButton': '⏳ Έλεγχος...',
    'options.diagnostics.copyButton': '📋 Αντιγραφή αναφοράς',
    'options.diagnostics.copyFirst': 'Πρώτα τρέξε τον έλεγχο.',
    'options.diagnostics.copySuccess': '✅ Η αναφορά αντιγράφηκε στο clipboard.',
    'options.diagnostics.copyFailed': 'Αποτυχία αντιγραφής.',
    'options.diagnostics.clipboardNotSupported': 'Το clipboard δεν υποστηρίζεται στο browser σου.',
    'options.diagnostics.checkError': 'Σφάλμα κατά τον έλεγχο',
    'options.diagnostics.reportTitle': 'LinguaLive Pro – Διαγνωστικά',
    'options.diagnostics.results': 'Αποτελέσματα',

    // Options: Diagnostic result names
    'options.diagnostics.result.environment': 'Περιβάλλον',
    'options.diagnostics.result.backend': 'Backend',
    'options.diagnostics.result.auth': 'Σύνδεση',
    'options.diagnostics.result.microphone': 'Μικρόφωνο',

    // Options: Diagnostic messages
    'options.diagnostics.env.ok': 'Οκ, το extension τρέχει σωστά στον browser σου.',
    'options.diagnostics.env.chromeApiMissing': 'Το Chrome API δεν είναι διαθέσιμο.',
    'options.diagnostics.env.missingFeatures': 'Λείπουν βασικές λειτουργίες browser (fetch/WebSocket).',
    'options.diagnostics.backend.ok': 'Ο server είναι προσβάσιμος.',
    'options.diagnostics.backend.unreachable': 'Δεν μπορέσαμε να επικοινωνήσουμε με τον server.',
    'options.diagnostics.backend.reachableAuthSeparate': 'Ο server είναι προσβάσιμος (auth ελέγχεται ξεχωριστά).',
    'options.diagnostics.backend.skipped': 'Παράλειψη (χρειάζεται πρώτα σύνδεση).',
    'options.diagnostics.auth.notConnected': 'Δεν έχεις συνδεθεί ακόμα.',
    'options.diagnostics.auth.connected': 'Συνδέθηκες ως',
    'options.diagnostics.auth.invalid': 'Το κλειδί/σύνδεση φαίνεται άκυρη ή ληγμένη.',
    'options.diagnostics.auth.unreachable': 'Δεν μπορέσαμε να επικοινωνήσουμε με τον server.',
    'options.diagnostics.mic.granted': 'Το μικρόφωνο είναι επιτρεπόμενο.',
    'options.diagnostics.mic.denied': 'Το μικρόφωνο είναι μπλοκαρισμένο. Άνοιξέ το από τις ρυθμίσεις του browser.',
    'options.diagnostics.mic.prompt': 'Θα σου ζητηθεί άδεια όταν ξεκινήσεις συνεδρία.',
    'options.diagnostics.mic.skipped': 'Παράλειψη (ο browser δεν υποστηρίζει έλεγχο αδειών μικροφώνου).',
    'options.diagnostics.mic.skippedBrowser': 'Παράλειψη (ο browser δεν υποστηρίζει έλεγχο αδειών).',

    // Options: Backend status diagnostics
    'options.diagnostics.result.backendStatus': 'Κατάσταση backend',
    'options.diagnostics.backendStatus.ok': 'OK',
    'options.diagnostics.backendStatus.unreachable': 'Το backend δεν είναι προσβάσιμο.',
    'options.diagnostics.backendStatus.error': 'Σφάλμα κατά τον έλεγχο του backend.',

    // Options: Realtime WebSocket diagnostics
    'options.diagnostics.result.realtimeWebSocket': 'Έλεγχος WebSocket (ζωντανή ροή)',
    'options.diagnostics.realtimeWs.ok': 'OK – η σύνδεση WebSocket λειτούργησε.',
    'options.diagnostics.realtimeWs.skippedNoAuth': 'Παράλειψη – δεν έχει ρυθμιστεί auth (dev token / χρήστης).',
    'options.diagnostics.realtimeWs.skippedNoBackend': 'Παράλειψη – δεν έχει ρυθμιστεί backend URL.',
    'options.diagnostics.realtimeWs.error': 'Σφάλμα κατά τη σύνδεση στο WebSocket.',

    // Options: Connection test
    'options.connection.testing': 'Σύνδεση…',
    'options.connection.button': 'Έλεγχος σύνδεσης',
    'options.connection.noAuth': '❌ Δεν έχεις ορίσει token ή debug user.',
    'options.connection.invalidToken': '❌ Μη έγκυρο token. Ελέγξτε τα στοιχεία σύνδεσης.',
    'options.connection.serverNotFound': '❌ Δεν βρέθηκε ο server. Ελέγξτε το Backend URL.',
    'options.connection.success': '✅ Συνδέθηκες ως',

    // Options: Generate dev token
    'options.backend.generateDevToken.button': 'Δημιουργία dev token',
    'options.backend.generateDevToken.generating': 'Δημιουργία…',
    'options.backend.generateDevToken.success': '✅ Νέο dev token δημιουργήθηκε και αποθηκεύτηκε.',
    'options.backend.generateDevToken.errorMissingUser': '❌ Ρύθμισε πρώτα το Debug user ID.',
    'options.backend.generateDevToken.errorMissingBackend': '❌ Ρύθμισε πρώτα το Backend URL.',
    'options.backend.generateDevToken.errorHttp': '❌ Αποτυχία δημιουργίας dev token (σφάλμα διακομιστή).',
    'options.backend.generateDevToken.errorNetwork': '❌ Αποτυχία δημιουργίας dev token (σφάλμα δικτύου).',

    // Options: Dev token expiry display
    'options.backend.devTokenExpiry.label': 'Λήξη dev token:',
    'options.backend.devTokenExpiry.none': 'Δεν υπάρχει ενεργό dev token.',

    // Options: Revoke dev token
    'options.backend.revokeDevToken.button': 'Ανάκληση dev token',
    'options.backend.revokeDevToken.revoking': 'Ανάκληση…',
    'options.backend.revokeDevToken.success': '✅ Το dev token ανακλήθηκε και αφαιρέθηκε από τις ρυθμίσεις.',
    'options.backend.revokeDevToken.errorNoToken': '❌ Δεν υπάρχει ενεργό dev token για ανάκληση.',
    'options.backend.revokeDevToken.errorHttp': '❌ Αποτυχία ανάκλησης dev token (σφάλμα διακομιστή).',
    'options.backend.revokeDevToken.errorNetwork': '❌ Αποτυχία ανάκλησης dev token (σφάλμα δικτύου).',

    // Options: Summary panel
    'options.summary.title': 'Λογαριασμός & Χρήση',
    'options.summary.plan': 'Πλάνο',
    'options.summary.usage': 'Χρήση',
    'options.summary.loading': 'Φόρτωση…',
    'options.summary.loadError': 'Δεν ήταν δυνατή η φόρτωση του πλάνου/χρήσης.',
    'options.summary.noAuth': 'Συνδέσου πρώτα στην LinguaLive για να δεις το πλάνο και τη χρήση σου.',
    'options.summary.minutes': 'λεπτά',
    'options.summary.sessions': 'συνεδρίες',
    'options.summary.sessionSingular': 'συνεδρία',
    'options.summary.perSession': '(ανά συνεδρία)',
    'options.summary.maxPerSession': 'Max',

    // Options: Language selector
    'options.language.label': 'Γλώσσα διεπαφής / Interface language',
    'options.language.auto': 'Αυτόματη (browser)',
    'options.language.el': 'Ελληνικά',
    'options.language.en': 'English',

    // Options: Button labels
    'options.buttons.testConnection': 'Έλεγχος σύνδεσης',
    'options.buttons.copyDiagnostics': '📋 Αντιγραφή αναφοράς',

    // Options: Shortcuts & controls section
    'options.shortcuts.sectionTitle': 'Συντομεύσεις & χειριστήρια',
    'options.shortcuts.howToStart': 'Πώς ξεκινάς τους υπότιτλους',
    'options.shortcuts.howToStart.steps': 'Κάνε κλικ στο εικονίδιο του LinguaLive Pro στη γραμμή εργαλείων ή χρησιμοποίησε τη συντόμευση πληκτρολογίου ή το μενού δεξιού κλικ.',
    'options.shortcuts.keyboardHeading': 'Συντομεύσεις πληκτρολογίου',
    'options.shortcuts.keyboardToggle': 'Εναλλαγή υποτίτλων: Alt+Shift+C',
    'options.shortcuts.keyboardOpenOptions': 'Άνοιγμα ρυθμίσεων: Alt+Shift+O',
    'options.shortcuts.contextMenuHeading': 'Μενού δεξιού κλικ',
    'options.shortcuts.contextMenuDescription': 'Κάνε δεξί κλικ σε σελίδα ή βίντεο και επίλεξε «Εναλλαγή υποτίτλων LinguaLive».',
    'options.shortcuts.chromeShortcutsNote': 'Μπορείς να προσαρμόσεις αυτές τις συντομεύσεις από τις ρυθμίσεις του Chrome: άνοιξε τη διεύθυνση chrome://extensions/shortcuts και βρες το «LinguaLive Pro».',

    // Options: Help & troubleshooting section
    'options.help.sectionTitle': 'Βοήθεια & αντιμετώπιση προβλημάτων',
    'options.help.whenToUseDiagnostics': 'Αν οι υπότιτλοι δεν ξεκινούν, αποσυνδέονται συνέχεια ή κάτι φαίνεται λάθος, χρησιμοποίησε πρώτα τα διαγνωστικά αντί να αλλάζεις τυχαία ρυθμίσεις.',
    'options.help.stepsHeading': 'Προτεινόμενα βήματα',
    'options.help.stepCheckBackend': '1. Τρέξε τα διαγνωστικά και έλεγξε ότι οι έλεγχοι backend και WebSocket είναι OK.',
    'options.help.stepCheckMic': '2. Έλεγξε τον έλεγχο μικροφώνου και βεβαιώσου ότι το browser έχει άδεια για τη σωστή συσκευή εισόδου.',
    'options.help.stepCopyReport': '3. Χρησιμοποίησε το «Αντιγραφή διαγνωστικών στο πρόχειρο» και επικόλλησε το report όταν επικοινωνείς με την υποστήριξη του LinguaLive Pro.',
    'options.help.privacyNote': 'Το διαγνωστικό report περιλαμβάνει τεχνικές πληροφορίες (plan, browser, κατάσταση backend) αλλά ποτέ το ηχητικό σου ή το κείμενο των υποτίτλων.',

    // Options: General
    'options.error.generic': 'Παρουσιάστηκε σφάλμα.',
  },

  en: {
    // HUD: Title & Description
    'hud.title': 'LinguaLive Pro',
    'hud.description': 'Real-time speech-to-text captions and translation overlay',

    // HUD: Status strings
    'hud.status.idle': 'Idle',
    'hud.status.connecting': 'Connecting…',
    'hud.status.connected': 'Connected',
    'hud.status.sessionStarted': 'Session started',
    'hud.status.live': 'Live captions',
    'hud.status.stopping': 'Stopping…',
    'hud.status.disconnected': 'Disconnected',
    'hud.status.error': 'Error – open diagnostics from the options page for more details.',
    'hud.status.noAuth': 'Not connected',
    'hud.status.authError': 'Authentication error',
    'hud.status.usageLimit': 'Usage limit',
    'hud.status.planLimit': 'Plan limit',
    'hud.status.usingFastMode': 'Using FAST mode',
    'hud.status.loadingLimits': 'Loading limits…',
    'hud.status.loadingUsage': 'Usage: …',

    // HUD: Auth required
    'hud.auth.required': 'To use LinguaLive, please connect your account first.',
    'hud.auth.tokenExpired': 'Your token is invalid or expired. Open settings to refresh it.',
    'hud.auth.openSettings': 'Open Settings',

    // HUD: Limits exceeded
    'hud.limits.reached': 'You\'ve reached your plan limit for today. Captions stopped.',
    'hud.limits.sessionReached': 'You\'ve reached your plan limit for this session. Captions stopped.',
    'hud.limits.rateLimited': 'Usage limit exceeded. Wait a bit or upgrade your plan.',
    'hud.limits.upgrade': 'Upgrade',
    'hud.limits.startButtonTitle': 'Limit exceeded – upgrade to continue',

    // HUD: Diagnostics link
    'hud.diagnostics.link': 'Diagnose problem',

    // HUD: Smart mode warning
    'hud.smartWarning.notAvailable': '⚠️ SMART mode not available.',
    'hud.smartWarning.upgradeHint': ' Upgrade for SMART mode.',

    // HUD: Transcript placeholder
    'hud.transcript.placeholder': 'Captions will appear here…',

    // HUD: Upgrade nudge
    'hud.upgrade.link': 'Upgrade',

    // HUD: Usage display
    'hud.usage.label': 'Usage',
    'hud.usage.minutes': 'min',
    'hud.usage.sessions': 'sessions',
    'hud.usage.sessionSingular': 'session',

    // Options: Diagnostics section
    'options.diagnostics.title': 'Diagnostics / System Check',
    'options.diagnostics.description': 'Run a quick check to see if everything is set up correctly (backend, connection, microphone).',
    'options.diagnostics.runButton': '🔍 Quick check',
    'options.diagnostics.runningButton': '⏳ Checking...',
    'options.diagnostics.copyButton': '📋 Copy report',
    'options.diagnostics.copyFirst': 'Run the check first.',
    'options.diagnostics.copySuccess': '✅ Report copied to clipboard.',
    'options.diagnostics.copyFailed': 'Failed to copy.',
    'options.diagnostics.clipboardNotSupported': 'Clipboard is not supported in your browser.',
    'options.diagnostics.checkError': 'Error during check',
    'options.diagnostics.reportTitle': 'LinguaLive Pro – Diagnostics',
    'options.diagnostics.results': 'Results',

    // Options: Diagnostic result names
    'options.diagnostics.result.environment': 'Environment',
    'options.diagnostics.result.backend': 'Backend',
    'options.diagnostics.result.auth': 'Connection',
    'options.diagnostics.result.microphone': 'Microphone',

    // Options: Diagnostic messages
    'options.diagnostics.env.ok': 'OK, the extension is running correctly in your browser.',
    'options.diagnostics.env.chromeApiMissing': 'Chrome API is not available.',
    'options.diagnostics.env.missingFeatures': 'Missing basic browser features (fetch/WebSocket).',
    'options.diagnostics.backend.ok': 'Server is reachable.',
    'options.diagnostics.backend.unreachable': 'Could not communicate with the server.',
    'options.diagnostics.backend.reachableAuthSeparate': 'Server is reachable (auth checked separately).',
    'options.diagnostics.backend.skipped': 'Skipped (connection required first).',
    'options.diagnostics.auth.notConnected': 'Not connected yet.',
    'options.diagnostics.auth.connected': 'Connected as',
    'options.diagnostics.auth.invalid': 'Token/connection appears invalid or expired.',
    'options.diagnostics.auth.unreachable': 'Could not communicate with the server.',
    'options.diagnostics.mic.granted': 'Microphone is allowed.',
    'options.diagnostics.mic.denied': 'Microphone is blocked. Enable it in browser settings.',
    'options.diagnostics.mic.prompt': 'Permission will be requested when you start a session.',
    'options.diagnostics.mic.skipped': 'Skipped (browser does not support microphone permission check).',
    'options.diagnostics.mic.skippedBrowser': 'Skipped (browser does not support permission check).',

    // Options: Backend status diagnostics
    'options.diagnostics.result.backendStatus': 'Backend status',
    'options.diagnostics.backendStatus.ok': 'OK',
    'options.diagnostics.backendStatus.unreachable': 'Backend is unreachable.',
    'options.diagnostics.backendStatus.error': 'Error while checking backend status.',

    // Options: Realtime WebSocket diagnostics
    'options.diagnostics.result.realtimeWebSocket': 'Realtime WebSocket check',
    'options.diagnostics.realtimeWs.ok': 'OK – WebSocket connection succeeded.',
    'options.diagnostics.realtimeWs.skippedNoAuth': 'Skipped – no auth configured (dev token / user).',
    'options.diagnostics.realtimeWs.skippedNoBackend': 'Skipped – no backend URL configured.',
    'options.diagnostics.realtimeWs.error': 'Error while connecting to WebSocket.',

    // Options: Connection test
    'options.connection.testing': 'Connecting…',
    'options.connection.button': 'Test connection',
    'options.connection.noAuth': '❌ No token or debug user set.',
    'options.connection.invalidToken': '❌ Invalid token. Check your credentials.',
    'options.connection.serverNotFound': '❌ Server not found. Check the Backend URL.',
    'options.connection.success': '✅ Connected as',

    // Options: Generate dev token
    'options.backend.generateDevToken.button': 'Generate dev token',
    'options.backend.generateDevToken.generating': 'Generating…',
    'options.backend.generateDevToken.success': '✅ New dev token generated and saved.',
    'options.backend.generateDevToken.errorMissingUser': '❌ Please configure Debug user ID first.',
    'options.backend.generateDevToken.errorMissingBackend': '❌ Please configure Backend URL first.',
    'options.backend.generateDevToken.errorHttp': '❌ Failed to generate dev token (server error).',
    'options.backend.generateDevToken.errorNetwork': '❌ Failed to generate dev token (network error).',

    // Options: Dev token expiry display
    'options.backend.devTokenExpiry.label': 'Dev token expires at:',
    'options.backend.devTokenExpiry.none': 'No active dev token.',

    // Options: Revoke dev token
    'options.backend.revokeDevToken.button': 'Revoke dev token',
    'options.backend.revokeDevToken.revoking': 'Revoking…',
    'options.backend.revokeDevToken.success': '✅ Dev token revoked and removed from settings.',
    'options.backend.revokeDevToken.errorNoToken': '❌ No active dev token to revoke.',
    'options.backend.revokeDevToken.errorHttp': '❌ Failed to revoke dev token (server error).',
    'options.backend.revokeDevToken.errorNetwork': '❌ Failed to revoke dev token (network error).',

    // Options: Summary panel
    'options.summary.title': 'Account & Usage',
    'options.summary.plan': 'Plan',
    'options.summary.usage': 'Usage',
    'options.summary.loading': 'Loading…',
    'options.summary.loadError': 'Could not load plan/usage.',
    'options.summary.noAuth': 'Connect to LinguaLive first to see your plan and usage.',
    'options.summary.minutes': 'minutes',
    'options.summary.sessions': 'sessions',
    'options.summary.sessionSingular': 'session',
    'options.summary.perSession': '(per session)',
    'options.summary.maxPerSession': 'Max',

    // Options: Language selector
    'options.language.label': 'Interface language',
    'options.language.auto': 'Auto (browser)',
    'options.language.el': 'Ελληνικά',
    'options.language.en': 'English',

    // Options: Button labels
    'options.buttons.testConnection': 'Test connection',
    'options.buttons.copyDiagnostics': '📋 Copy report',

    // Options: Shortcuts & controls section
    'options.shortcuts.sectionTitle': 'Shortcuts & controls',
    'options.shortcuts.howToStart': 'How to start captions',
    'options.shortcuts.howToStart.steps': 'Click the LinguaLive Pro icon in the toolbar or use the keyboard shortcut or context menu.',
    'options.shortcuts.keyboardHeading': 'Keyboard shortcuts',
    'options.shortcuts.keyboardToggle': 'Toggle captions: Alt+Shift+C',
    'options.shortcuts.keyboardOpenOptions': 'Open settings: Alt+Shift+O',
    'options.shortcuts.contextMenuHeading': 'Context menu',
    'options.shortcuts.contextMenuDescription': 'Right-click on a page or video and choose "Toggle LinguaLive captions".',
    'options.shortcuts.chromeShortcutsNote': 'You can customize these shortcuts from Chrome\'s settings: open chrome://extensions/shortcuts and look for "LinguaLive Pro".',

    // Options: Help & troubleshooting section
    'options.help.sectionTitle': 'Help & troubleshooting',
    'options.help.whenToUseDiagnostics': 'If captions don\'t start, keep disconnecting, or something feels off, use the diagnostics before changing settings at random.',
    'options.help.stepsHeading': 'Recommended steps',
    'options.help.stepCheckBackend': '1. Run diagnostics and check that the backend status and realtime WebSocket checks are OK.',
    'options.help.stepCheckMic': '2. Check the microphone diagnostic and make sure your browser has permission to use the correct input device.',
    'options.help.stepCopyReport': '3. Use "Copy diagnostics to clipboard" and paste the report when contacting LinguaLive Pro support.',
    'options.help.privacyNote': 'The diagnostics report includes technical environment info (plan, browser, backend status) but never your audio or transcript content.',

    // Options: General
    'options.error.generic': 'An error occurred.',
  },
};

// =============================================================================
// LOCALE MANAGEMENT
// =============================================================================

/**
 * Current active locale.
 * Defaults to Greek ('el') as the primary language.
 */
let currentLocale: Locale = 'el';

/**
 * Detect locale from browser navigator.language.
 * Falls back to 'el' (Greek) if detection fails or language is not supported.
 *
 * @returns Detected locale
 */
function detectLocale(): Locale {
  // Guard for environments where navigator is not available (e.g., tests, bundler)
  if (typeof navigator === 'undefined' || !navigator.language) {
    return 'el';
  }

  const lang = navigator.language.toLowerCase();

  // Greek detection
  if (lang.startsWith('el')) {
    return 'el';
  }

  // English detection (or any other language falls back to English)
  if (lang.startsWith('en')) {
    return 'en';
  }

  // Default fallback to Greek (primary language)
  return 'el';
}

/**
 * Set the current locale.
 * Use this to programmatically change the language.
 *
 * @param locale - The locale to set
 */
export function setLocale(locale: Locale): void {
  currentLocale = locale;
}

/**
 * Get the current locale.
 *
 * @returns The current active locale
 */
export function getLocale(): Locale {
  return currentLocale;
}

// Initialize locale on module load (if navigator is available)
if (typeof navigator !== 'undefined') {
  currentLocale = detectLocale();
}

// =============================================================================
// TRANSLATION FUNCTION
// =============================================================================

/**
 * Get a translated message for the current locale.
 *
 * Lookup order:
 * 1. Current locale's message
 * 2. Greek (primary) fallback
 * 3. Message key itself (if no translation found)
 *
 * @param key - The message key to look up
 * @returns The translated message string
 */
export function t(key: MessageKey): string {
  // Try current locale
  const table = MESSAGES[currentLocale];
  if (table && table[key]) {
    return table[key];
  }

  // Fallback to Greek (primary language)
  const fallback = MESSAGES.el;
  if (fallback && fallback[key]) {
    return fallback[key];
  }

  // Last resort: return the key itself
  return key;
}
