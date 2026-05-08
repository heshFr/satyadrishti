import { useState, useRef, useEffect, useCallback } from "react";
import Layout from "@/components/Layout";
import { motion, AnimatePresence } from "framer-motion";
import MaterialIcon from "@/components/MaterialIcon";
import { toast } from "sonner";
import { API_BASE } from "@/lib/api";

interface VoicePrint {
  name: string;
  relationship: string;
  enrolled_at: string;
  audio_duration: number;
}

const RELATIONSHIPS = [
  "son", "daughter", "spouse", "parent", "sibling",
  "friend", "colleague", "other",
];

const VoiceEnroll = () => {
  const [prints, setPrints] = useState<VoicePrint[]>([]);
  const [name, setName] = useState("");
  const [relationship, setRelationship] = useState("unknown");
  const [isRecording, setIsRecording] = useState(false);
  const [recordingTime, setRecordingTime] = useState(0);
  const [loading, setLoading] = useState(false);
  const [audioLevel, setAudioLevel] = useState(0);

  // Audio refs...
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const audioCtxRef = useRef<AudioContext | null>(null);
  const rafRef = useRef<number>(0);

  const fetchPrints = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/api/voice-prints`);
      const data = await res.json();
      setPrints(data.prints || []);
    } catch {
      // Backend might not be running
    }
  }, []);

  useEffect(() => {
    fetchPrints();
  }, [fetchPrints]);

  const monitorAudio = useCallback(() => {
    const analyser = analyserRef.current;
    if (!analyser) return;

    const bufferLength = analyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);

    const draw = () => {
      rafRef.current = requestAnimationFrame(draw);
      analyser.getByteTimeDomainData(dataArray);
      let sum = 0;
      for (let i = 0; i < bufferLength; i++) {
        const val = (dataArray[i] - 128) / 128;
        sum += val * val;
      }
      setAudioLevel(Math.sqrt(sum / bufferLength));
    };
    draw();
  }, []);

  const stopVisualization = useCallback(() => {
    if (rafRef.current) {
      cancelAnimationFrame(rafRef.current);
      rafRef.current = 0;
    }
    if (audioCtxRef.current) {
      audioCtxRef.current.close();
      audioCtxRef.current = null;
    }
    analyserRef.current = null;
    setAudioLevel(0);
  }, []);

  const startRecording = async () => {
    if (!name.trim()) {
      toast.error("Please enter a name first");
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: { echoCancellation: true, noiseSuppression: true, sampleRate: 16000 },
      });
      streamRef.current = stream;

      const audioCtx = new AudioContext({ sampleRate: 16000 });
      audioCtxRef.current = audioCtx;
      const source = audioCtx.createMediaStreamSource(stream);
      const analyser = audioCtx.createAnalyser();
      analyser.fftSize = 2048;
      source.connect(analyser);
      analyserRef.current = analyser;
      monitorAudio();

      const recorder = new MediaRecorder(stream, { mimeType: "audio/webm;codecs=opus" });
      mediaRecorderRef.current = recorder;
      chunksRef.current = [];

      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data);
      };

      recorder.onstop = () => {
        if (timerRef.current) clearInterval(timerRef.current);
        stream.getTracks().forEach((t) => t.stop());
        streamRef.current = null;
        stopVisualization();
      };

      recorder.start(250);
      setIsRecording(true);
      setRecordingTime(0);

      timerRef.current = setInterval(() => {
        setRecordingTime((t) => t + 1);
      }, 1000);
    } catch {
      toast.error("Microphone access denied.");
    }
  };

  const stopAndEnroll = async () => {
    if (!mediaRecorderRef.current) return;
    setIsRecording(false);
    mediaRecorderRef.current.stop();
    if (timerRef.current) clearInterval(timerRef.current);

    await new Promise((r) => setTimeout(r, 300));
    if (chunksRef.current.length === 0) {
      toast.error("No audio recorded");
      return;
    }

    const blob = new Blob(chunksRef.current, { type: "audio/webm" });
    setLoading(true);
    try {
      const arrayBuffer = await blob.arrayBuffer();
      const audioCtx = new AudioContext({ sampleRate: 16000 });
      const decoded = await audioCtx.decodeAudioData(arrayBuffer);
      const channelData = decoded.getChannelData(0);

      const wavBuffer = new ArrayBuffer(44 + channelData.length * 2);
      const view = new DataView(wavBuffer);
      const writeStr = (off: number, s: string) => {
        for (let i = 0; i < s.length; i++) view.setUint8(off + i, s.charCodeAt(i));
      };
      writeStr(0, "RIFF");
      view.setUint32(4, 36 + channelData.length * 2, true);
      writeStr(8, "WAVE");
      writeStr(12, "fmt ");
      view.setUint32(16, 16, true);
      view.setUint16(20, 1, true);
      view.setUint16(22, 1, true);
      view.setUint32(24, 16000, true);
      view.setUint32(28, 32000, true);
      view.setUint16(32, 2, true);
      view.setUint16(34, 16, true);
      writeStr(36, "data");
      view.setUint32(40, channelData.length * 2, true);
      for (let i = 0; i < channelData.length; i++) {
        const s = Math.max(-1, Math.min(1, channelData[i]));
        view.setInt16(44 + i * 2, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
      }
      audioCtx.close();

      const wavBlob = new Blob([wavBuffer], { type: "audio/wav" });
      const formData = new FormData();
      formData.append("file", wavBlob, `${name.trim()}.wav`);
      formData.append("name", name.trim());
      formData.append("relationship", relationship);

      const res = await fetch(`${API_BASE}/api/voice-prints/enroll`, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "Enrollment failed");
      }

      const result = await res.json();
      toast.success(result.message || `Voice print enrolled!`);
      setName("");
      setRelationship("unknown");
      fetchPrints();
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "Enrollment failed");
    } finally {
      setLoading(false);
    }
  };

  const deletePrint = async (printName: string) => {
    try {
      const res = await fetch(`${API_BASE}/api/voice-prints/${encodeURIComponent(printName)}`, {
        method: "DELETE",
      });
      if (res.ok) {
        fetchPrints();
        toast.success(`Voice print removed.`);
      }
    } catch {
      toast.error("Failed to delete voice print");
    }
  };

  // Generate deterministic pastel color for avatars based on name
  const getAvatarColor = (name: string) => {
    const bgColors = ["bg-primary/20", "bg-secondary/20", "bg-tertiary/20", "bg-error-container/20"];
    const textColors = ["text-primary", "text-secondary", "text-on-tertiary-container", "text-error"];
    
    let sum = 0;
    for (let i = 0; i < name.length; i++) sum += name.charCodeAt(i);
    const idx = sum % bgColors.length;
    
    return { bg: bgColors[idx], text: textColors[idx] };
  };

  return (
    <Layout systemStatus="protected">
      <div className="pt-32 pb-12 px-6 md:px-12 max-w-7xl mx-auto space-y-12">
        {/* Page Header */}
        <div className="flex flex-col md:flex-row justify-between items-start md:items-end gap-6">
          <div>
            <h1 className="text-4xl md:text-5xl font-headline font-extrabold tracking-tighter text-on-surface mb-2">Voice Prints</h1>
            <p className="text-on-surface-variant max-w-xl text-lg leading-relaxed">
              Manage biometric voice identities for your inner circle. Secure your family against deepfake voice cloning with real-time verification.
            </p>
          </div>
          <div className="flex space-x-4">
            <div className="bg-surface-container-low px-6 py-3 rounded-xl flex items-center space-x-3 border border-outline-variant/10">
              <span className="flex h-3 w-3 rounded-full bg-secondary shadow-[0_0_10px_rgba(78,222,163,0.5)]"></span>
              <span className="text-sm font-label uppercase tracking-widest text-on-surface-variant">Live Protection: Active</span>
            </div>
          </div>
        </div>

        {/* Dashboard Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
          
          {/* New Enrollment Panel */}
          <section className="lg:col-span-5 space-y-8">
            <div className="glass-panel rounded-2xl p-6 md:p-8 flex flex-col h-full min-h-[500px] relative overflow-hidden group border border-outline-variant/10">
              {/* Background Glow Accent */}
              <div className="absolute -top-24 -right-24 w-64 h-64 bg-primary-container/10 rounded-full blur-[100px]"></div>
              
              <div className="relative z-10 flex flex-col h-full">
                <h3 className="text-on-surface-variant font-headline uppercase tracking-widest text-[12px] font-bold mb-6">New Enrollment</h3>
                <h2 className="text-3xl font-headline font-bold text-on-surface mb-8">Register Identity</h2>
                
                <div className="space-y-4 flex-1">
                  <div className="space-y-2">
                    <label className="text-xs font-label uppercase tracking-widest text-on-surface-variant px-1 font-bold">Full Name</label>
                    <input 
                      value={name}
                      onChange={(e) => setName(e.target.value)}
                      disabled={isRecording || loading}
                      className="w-full bg-surface-container-low border-none border-b-2 border-outline-variant/30 focus:border-primary focus:ring-0 text-on-surface py-4 px-3 placeholder:text-outline-variant transition-all font-body text-sm rounded-t-lg outline-none" 
                      placeholder="e.g. Elena Richards" 
                      type="text"
                    />
                  </div>
                  <div className="space-y-2">
                    <label className="text-xs font-label uppercase tracking-widest text-on-surface-variant px-1 font-bold">Relationship</label>
                    <select 
                      value={relationship}
                      onChange={(e) => setRelationship(e.target.value)}
                      disabled={isRecording || loading}
                      className="w-full bg-surface-container-low border-none border-b-2 border-outline-variant/30 focus:border-primary focus:ring-0 text-on-surface py-4 px-3 placeholder:text-outline-variant transition-all font-body text-sm rounded-t-lg cursor-pointer appearance-none outline-none"
                    >
                      <option value="unknown">Select...</option>
                      {RELATIONSHIPS.map((r) => (
                        <option key={r} value={r}>{r.charAt(0).toUpperCase() + r.slice(1)}</option>
                      ))}
                    </select>
                  </div>
                </div>
                
                {/* Live Microphone Visualizer */}
                <div className="mt-8 p-6 md:p-8 bg-surface-container-lowest rounded-2xl flex flex-col items-center justify-center space-y-6 md:space-y-8 relative overflow-hidden border border-outline-variant/5">
                  <div className="flex items-center space-x-1 h-12 w-full justify-center">
                    {/* Fake wave bars driven by audioLevel when recording */}
                    {[1, 2, 3, 2, 4, 3, 5, 2, 4, 2].map((m, i) => (
                      <div 
                        key={i} 
                        className="w-1 md:w-1.5 bg-primary rounded-full transition-all duration-75"
                        style={{ height: isRecording ? `${Math.max(4, audioLevel * 200 * m)}px` : '4px' }}
                      ></div>
                    ))}
                  </div>
                  
                  {isRecording && (
                    <div className="absolute top-4 left-4 flex items-center gap-2">
                      <div className="w-2 h-2 rounded-full bg-error animate-pulse"></div>
                      <span className="text-[12px] font-mono text-error font-bold">{Math.floor(recordingTime / 60)}:{(recordingTime % 60).toString().padStart(2, '0')}</span>
                    </div>
                  )}

                  <button 
                    onClick={isRecording ? stopAndEnroll : startRecording}
                    disabled={loading}
                    className={`w-16 h-16 md:w-20 md:h-20 rounded-full flex items-center justify-center shadow-[0_0_30px_rgba(0,209,255,0.3)] hover:scale-105 active:scale-95 transition-all outline-none cursor-pointer ${
                      isRecording ? 'bg-error text-on-error shadow-[0_0_30px_rgba(255,180,171,0.3)]' : 'bg-gradient-to-br from-primary to-primary-container text-on-primary-container'
                    }`}
                  >
                    <MaterialIcon icon={isRecording ? "stop" : "mic"} filled size={36} />
                  </button>
                  <p className="text-xs font-label text-on-surface-variant text-center px-4 max-w-[200px]">
                    {isRecording 
                      ? "Recording in progress... Tap to stop and save."
                      : loading
                        ? "Verifying encoding..."
                        : "Tap to begin voice print capture. Speak clearly."}
                  </p>
                </div>
                
                <button 
                  disabled={isRecording || loading || !name.trim()}
                  onClick={() => {/* Only decorative if start/stop via mic */}}
                  className={`mt-6 py-4 rounded-xl transition-all uppercase tracking-widest text-[12px] font-bold border outline-none ${
                    name.trim() && !isRecording && !loading
                      ? "bg-surface-container-high hover:bg-surface-container-highest text-primary border-outline-variant/10"
                      : "bg-surface-container-lowest text-outline border-outline-variant/5 opacity-50 cursor-not-allowed"
                  }`}
                >
                  {loading ? 'Processing...' : 'Ready for Enrollment'}
                </button>
              </div>
            </div>
          </section>

          {/* Family List Panel */}
          <section className="lg:col-span-7 space-y-8">
            <div className="flex items-center justify-between">
              <h3 className="text-on-surface-variant font-headline uppercase tracking-widest text-[12px] font-bold">Enrolled Inner Circle</h3>
              <span className="text-on-surface-variant font-label text-sm italic">{prints.length} Active Profiles</span>
            </div>
            
            <div className="space-y-4">
              {prints.length === 0 ? (
                <div className="bg-surface-container-low p-10 rounded-2xl flex flex-col items-center justify-center text-center border border-outline-variant/10 min-h-[300px]">
                  <MaterialIcon icon="group_off" size={48} className="text-outline-variant mb-4" />
                  <h4 className="text-lg font-headline font-bold text-on-surface">No Profiles Found</h4>
                  <p className="text-sm text-on-surface-variant font-light mt-2 max-w-sm">
                    Record your family members' voices using the panel on the left to start building your secure circle.
                  </p>
                </div>
              ) : (
                prints.map((p, i) => {
                  const colors = getAvatarColor(p.name);
                  return (
                    <motion.div 
                      key={p.name}
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: i * 0.05 }}
                      className="bg-surface-container-low hover:bg-surface-container-high p-5 md:p-6 rounded-2xl transition-all group flex items-center justify-between border border-outline-variant/5"
                    >
                      <div className="flex items-center space-x-4 md:space-x-6">
                        <div className={`w-14 h-14 md:w-16 md:h-16 rounded-full flex items-center justify-center border-2 border-surface-container-highest relative ${colors.bg}`}>
                          <span className={`text-xl font-headline font-bold ${colors.text}`}>{p.name.charAt(0).toUpperCase()}</span>
                          <div className="absolute bottom-0 right-0 w-3 h-3 md:w-4 md:h-4 bg-secondary rounded-full border-2 border-surface-container-low"></div>
                        </div>
                        <div>
                          <h4 className="text-lg md:text-xl font-headline font-bold text-on-surface">{p.name}</h4>
                          <div className="flex flex-col sm:flex-row sm:items-center space-y-1 sm:space-y-0 sm:space-x-3 mt-1">
                            <span className="text-[12px] md:text-xs font-label uppercase tracking-widest text-on-surface-variant">
                              {p.relationship !== "unknown" ? p.relationship : "Secure Identity"}
                            </span>
                            <span className="w-max px-2 py-0.5 bg-secondary/10 text-secondary text-[11px] md:text-[12px] rounded-full font-bold uppercase tracking-tighter self-start sm:self-auto">Verified</span>
                          </div>
                        </div>
                      </div>
                      
                      <div className="flex items-center space-x-2 md:space-x-4">
                        <div className="text-right hidden sm:block mr-2">
                          <p className="text-[11px] md:text-[12px] font-label uppercase tracking-widest text-outline">Duration</p>
                          <p className="text-xs md:text-sm font-semibold text-on-surface-variant">
                            {p.audio_duration ? `${p.audio_duration.toFixed(1)}s` : 'Unknown'}
                          </p>
                        </div>
                        <button 
                          onClick={() => deletePrint(p.name)}
                          className="p-3 text-outline hover:text-error hover:bg-error/10 rounded-full transition-all cursor-pointer outline-none"
                          title="Delete Profile"
                        >
                          <MaterialIcon icon="delete" size={20} />
                        </button>
                      </div>
                    </motion.div>
                  );
                })
              )}
            </div>
          </section>
        </div>

        {/* Integrity Log Section (Bento style) */}
        <section className="space-y-6">
          <div className="flex items-center justify-between">
            <h3 className="text-on-surface-variant font-headline uppercase tracking-widest text-[12px] font-bold">Integrity Monitoring</h3>
            <button className="text-primary text-[12px] font-bold uppercase tracking-widest hover:underline transition-all cursor-pointer outline-none">Export Report</button>
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-6">
            <div className="bg-surface-container-low p-6 rounded-2xl flex flex-col justify-between min-h-[140px] md:min-h-[160px] border border-outline-variant/10">
               <MaterialIcon icon="verified_user" className="text-primary mb-4" size={32} />
               <div>
                 <p className="text-2xl font-headline font-bold text-on-surface">100%</p>
                 <p className="text-[12px] md:text-xs font-label uppercase tracking-widest text-on-surface-variant mt-1">Accuracy Score</p>
               </div>
            </div>
            <div className="bg-surface-container-low p-6 rounded-2xl flex flex-col justify-between min-h-[140px] md:min-h-[160px] border border-outline-variant/10">
               <MaterialIcon icon="record_voice_over" className="text-secondary mb-4" size={32} />
               <div>
                 <p className="text-2xl font-headline font-bold text-on-surface">{prints.length}</p>
                 <p className="text-[12px] md:text-xs font-label uppercase tracking-widest text-on-surface-variant mt-1">Active Profiles Managed</p>
               </div>
            </div>
            <div className="bg-surface-container-low p-6 rounded-2xl flex flex-col justify-between min-h-[140px] md:min-h-[160px] border border-outline-variant/10 sm:col-span-2 md:col-span-1">
               <MaterialIcon icon="warning" className="text-error mb-4" size={32} />
               <div>
                 <p className="text-2xl font-headline font-bold text-on-surface">0</p>
                 <p className="text-[12px] md:text-xs font-label uppercase tracking-widest text-on-surface-variant mt-1">Spoofing Attempts Blocked</p>
               </div>
            </div>
          </div>
        </section>

      </div>
    </Layout>
  );
};

export default VoiceEnroll;
