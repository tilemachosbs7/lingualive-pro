# Variable Playback Speed Support

## Επισκόπηση

Η εφαρμογή τώρα υποστηρίζει **αυτόματη προσαρμογή** σε διαφορετικές ταχύτητες αναπαραγωγής (0.5x - 2x) χωρίς latency ή απώλεια περιεχομένου.

## Πώς Λειτουργεί

### 1. **Auto-Detection (Frontend)**
- Το HUD ανιχνεύει αυτόματα το `video.playbackRate` από τη σελίδα
- Στέλνει την ταχύτητα στο backend κατά τη σύνδεση
- Παρακολουθεί για αλλαγές κάθε 500ms και ενημερώνει το backend δυναμικά

### 2. **Adaptive Thresholds (Backend)**

#### **Speed Mode** (≥ 1.5x)
- **Trigger**: Όταν το video παίζει σε 1.5x ή 2x ταχύτητα
- **Emergency Flush**: 3 δευτερόλεπτα (vs 5s κανονικά)
- **Translation Mode**: Fast-only (χωρίς refine pass για χαμηλότερο latency)
- **Σκοπός**: Αποφυγή υστέρησης σε γρήγορα videos

#### **Quality Mode** (≤ 0.75x)
- **Trigger**: Όταν το video παίζει σε 0.5x ή 0.75x ταχύτητα
- **Emergency Flush**: 7 δευτερόλεπτα (περισσότερος χρόνος για refine)
- **Translation Mode**: Full two-pass (fast + refine)
- **Σκοπός**: Μέγιστη ποιότητα σε αργά videos

#### **Normal Mode** (0.75x - 1.5x)
- **Emergency Flush**: 5 δευτερόλεπτα (default)
- **Translation Mode**: Two-pass με balanced thresholds

### 3. **Dynamic Updates**
- Το playback rate ενημερώνεται σε **real-time**
- Δεν χρειάζεται restart - αλλάζει αυτόματα
- Εμφανίζεται toast notification: "Speed: 2x"

## Τεχνικές Λεπτομέρειες

### Frontend Changes ([hud.ts](extension/src/content/hud.ts))

1. **`detectPlaybackRate()`**: Βρίσκει video elements και επιστρέφει playback rate
2. **`startPlaybackRateMonitoring()`**: Interval που ελέγχει για αλλαγές κάθε 500ms
3. **Config Message**: Προσθήκη `playbackRate` field
4. **Update Message**: Νέο `playback_rate_update` message type

```typescript
const config = {
  type: "config",
  sourceLang: "auto",
  targetLang: "en",
  playbackRate: 1.5,  // Αυτόματα detected
  // ...
};
```

### Backend Changes ([deepgram_realtime.py](backend/app/api/deepgram_realtime.py))

1. **Session Variable**: `playback_rate = 1.0`
2. **Config Handler**: Extract και log playback rate
3. **Adaptive Logic**:
```python
if playback_rate >= 1.5:
    emergency_flush_interval = 3.0  # Speed Mode
    speed_mode = True
elif playback_rate <= 0.75:
    emergency_flush_interval = 7.0  # Quality Mode
else:
    emergency_flush_interval = 5.0  # Normal
```
4. **Update Handler**: Νέο `playback_rate_update` message type

## Testing

### Έλεγχος με διαφορετικά speeds:

1. **0.5x Speed** (Slow Motion):
   - Άνοιξε YouTube video
   - Settings → Playback speed → 0.5
   - Έλεγξε logs: "Quality Mode (playback=0.5x): flush=7s"

2. **1x Speed** (Normal):
   - Default behavior
   - "Normal Mode (playback=1.0x): flush=5s"

3. **2x Speed** (Fast):
   - Settings → Playback speed → 2
   - Έλεγξε logs: "Speed Mode enabled (playback=2.0x): flush=3s"

### Αναμενόμενα Logs

**Frontend (Console)**:
```
[LinguaLive] Detected playback rate: 2x
[LinguaLive] Playback rate changed to 2x
```

**Backend (Terminal)**:
```
[abc123] Config: auto -> en, provider=deepl, quality=fast, 
         sampleRate=16000, playbackRate=2.0x, 
         endpointing=1000ms, flushInterval=3.0s
[abc123] Speed Mode enabled (playback=2.0x): flush=3s
```

## Troubleshooting

### Πρόβλημα: Δεν ανιχνεύει playback rate
- **Αιτία**: Το video element δεν υπάρχει ακόμα στο DOM
- **Λύση**: Το monitoring ξεκινά μετά το "ready" event, οπότε θα το πιάσει στο επόμενο tick

### Πρόβλημα: Latency σε 2x speed
- **Έλεγχος**: Δες τα logs για "Speed Mode enabled"
- **Αναμενόμενο**: p95 latency < 900ms με Speed Mode
- **Σημειώσεις**: Το Speed Mode παραλείπει refine pass για χαμηλότερο latency

### Πρόβλημα: Χαμηλή ποιότητα σε 0.5x speed
- **Έλεγχος**: Quality Mode πρέπει να ενεργοποιείται
- **Αναμενόμενο**: Full two-pass translation με 7s flush
- **Σημειώσεις**: Περισσότερος χρόνος = καλύτερες μεταφράσεις

## Μελλοντικές Βελτιώσεις

- [ ] Configurable thresholds ανά speed tier
- [ ] UI indicator για current speed mode
- [ ] Metrics για frame drops σε 2x
- [ ] Optional quality degradation με higher speeds
- [ ] Support για >2x speeds (3x, 4x)

## Status

✅ **Production Ready**
- Δουλεύει σε όλα τα speeds (0.5x - 2x)
- Auto-detection και dynamic updates
- Adaptive thresholds με optimized latency
- Tested και verified

---

**Last Updated**: 24 Δεκεμβρίου 2025  
**Related Files**: `extension/src/content/hud.ts`, `backend/app/api/deepgram_realtime.py`
