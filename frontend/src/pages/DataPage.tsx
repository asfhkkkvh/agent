import { useCallback, useEffect, useState } from "react"
import { toast } from "sonner"
import {
  BookOpenText,
  CloudUpload,
  FileText,
  Files,
  Loader2,
  Table2,
  Upload,
} from "lucide-react"
import { cn } from "@/lib/utils"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Textarea } from "@/components/ui/textarea"
import { getStats, ingestFile, ingestText, type CollectionStats } from "@/lib/api"

function StatCard({
  label,
  value,
  icon: Icon,
  tint,
}: {
  label: string
  value: string
  icon: typeof Files
  tint: string
}) {
  return (
    <Card className="border-border/60 bg-card/60 backdrop-blur-sm">
      <CardContent className="flex items-center gap-4 p-5">
        <div className={cn("flex h-11 w-11 items-center justify-center rounded-2xl", tint)}>
          <Icon className="h-5 w-5" />
        </div>
        <div>
          <div className="text-2xl font-semibold tabular-nums">{value}</div>
          <div className="text-xs text-muted-foreground">{label}</div>
        </div>
      </CardContent>
    </Card>
  )
}

function Dropzone({
  onUpload,
}: {
  onUpload: (files: File[]) => void
}) {
  const [drag, setDrag] = useState(false)
  const [busy, setBusy] = useState(false)

  const handleFiles = useCallback(
    async (list: FileList | File[]) => {
      const files = Array.from(list).filter((f) =>
        [".pdf", ".docx", ".txt", ".md"].some((s) => f.name.toLowerCase().endsWith(s)),
      )
      if (files.length === 0) {
        toast.warning("仅支持 PDF / DOCX / TXT / MD 文件")
        return
      }
      setBusy(true)
      onUpload(files)
      try {
        for (const file of files) {
          const res = await ingestFile(file)
          toast.success(`${file.name}:${res.message}`)
        }
        toast.success("导入完成")
      } catch (e) {
        toast.error(String(e))
      } finally {
        setBusy(false)
      }
    },
    [onUpload],
  )

  return (
    <label
      onDragOver={(e) => {
        e.preventDefault()
        setDrag(true)
      }}
      onDragLeave={() => setDrag(false)}
      onDrop={(e) => {
        e.preventDefault()
        setDrag(false)
        void handleFiles(e.dataTransfer.files)
      }}
      className={cn(
        "flex cursor-pointer flex-col items-center justify-center rounded-2xl border-2 border-dashed px-6 py-12 text-center transition-all",
        drag
          ? "border-primary/70 bg-primary/10"
          : "border-border/70 bg-card/40 hover:border-primary/40 hover:bg-card/60",
      )}
    >
      <input
        type="file"
        accept=".pdf,.docx,.txt,.md"
        multiple
        className="hidden"
        onChange={(e) => {
          if (e.target.files) void handleFiles(e.target.files)
          e.target.value = ""
        }}
      />
      <div className="mb-3 flex h-12 w-12 items-center justify-center rounded-2xl bg-gradient-to-br from-primary/80 to-cyan-500/80 shadow-lg shadow-primary/25">
        {busy ? <Loader2 className="h-6 w-6 animate-spin text-white" /> : <CloudUpload className="h-6 w-6 text-white" />}
      </div>
      <div className="text-sm font-medium">拖拽文件到此处,或点击选择</div>
      <div className="mt-1 text-xs text-muted-foreground">支持 PDF · DOCX · TXT · Markdown,自动分块 + 双路嵌入入库</div>
    </label>
  )
}

export function DataPage() {
  const [stats, setStats] = useState<CollectionStats | null>(null)
  const [text, setText] = useState("")
  const [source, setSource] = useState("manual")
  const [textBusy, setTextBusy] = useState(false)

  const refresh = useCallback(() => {
    getStats()
      .then(setStats)
      .catch(() => setStats({ points: null, sources: {}, content_types: {}, collection: "", error: "无法连接知识库" }))
  }, [])

  useEffect(refresh, [refresh])

  async function handleTextIngest() {
    if (!text.trim()) return
    setTextBusy(true)
    try {
      const res = await ingestText(text, source || "manual")
      toast.success(res.message)
      setText("")
      refresh()
    } catch (e) {
      toast.error(String(e))
    } finally {
      setTextBusy(false)
    }
  }

  const sourceCount = stats?.sources ? Object.keys(stats.sources).length : 0

  return (
    <div className="flex h-full flex-col overflow-hidden">
      <header className="border-b border-border/50 px-6 py-4 backdrop-blur-sm">
        <h1 className="text-lg font-semibold tracking-tight">知识库管理</h1>
        <p className="text-xs text-muted-foreground">导入文档 → Docling 解析 → 分块 → 稠密 + 稀疏双路嵌入 → Qdrant 混合索引</p>
      </header>

      <div className="min-h-0 flex-1 overflow-y-auto">
        <div className="mx-auto max-w-4xl space-y-6 px-6 py-6">
          {/* 统计 */}
          <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
            <StatCard label="向量分块" value={stats ? String(stats.points ?? "—") : "…"} icon={Files} tint="bg-primary/15 text-primary" />
            <StatCard label="来源文档" value={stats ? String(sourceCount) : "…"} icon={BookOpenText} tint="bg-cyan-500/15 text-cyan-400" />
            <StatCard label="文本块" value={stats ? String(stats.content_types?.text ?? 0) : "…"} icon={FileText} tint="bg-emerald-500/15 text-emerald-400" />
            <StatCard label="表格块" value={stats ? String(stats.content_types?.table ?? 0) : "…"} icon={Table2} tint="bg-violet-500/15 text-violet-400" />
          </div>

          {stats?.error && (
            <p className="rounded-xl border border-red-500/30 bg-red-500/10 px-4 py-3 text-sm text-red-300">
              {stats.error} —— 请确认 Qdrant 服务可用后再试。
            </p>
          )}

          {/* 上传 */}
          <Card className="border-border/60 bg-card/50 backdrop-blur-sm">
            <CardHeader>
              <CardTitle className="text-base">上传文件</CardTitle>
              <CardDescription>解析 PDF / DOCX / TXT / MD,自动去重、分块并双路嵌入入库</CardDescription>
            </CardHeader>
            <CardContent>
              <Dropzone onUpload={() => setTimeout(refresh, 3000)} />
            </CardContent>
          </Card>

          {/* 文本导入 */}
          <Card className="border-border/60 bg-card/50 backdrop-blur-sm">
            <CardHeader>
              <CardTitle className="text-base">直接导入文本</CardTitle>
              <CardDescription>粘贴文本片段快速入库,适合临时资料</CardDescription>
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="grid gap-3">
                <div className="space-y-1.5">
                  <Label htmlFor="source-label">来源标签</Label>
                  <Input id="source-label" value={source} onChange={(e) => setSource(e.target.value)} placeholder="例如: 季度财报" />
                </div>
                <div className="space-y-1.5">
                  <Label htmlFor="text-body">文本内容</Label>
                  <Textarea id="text-body" rows={5} value={text} onChange={(e) => setText(e.target.value)} placeholder="粘贴要导入的文本…" />
                </div>
              </div>
              <Button onClick={() => void handleTextIngest()} disabled={textBusy || !text.trim()}>
                {textBusy ? <Loader2 className="h-4 w-4 animate-spin" /> : <Upload className="h-4 w-4" />}
                导入文本
              </Button>
            </CardContent>
          </Card>

          {/* 来源分布 */}
          {stats && !stats.error && Object.keys(stats.sources ?? {}).length > 0 && (
            <Card className="border-border/60 bg-card/50 backdrop-blur-sm">
              <CardHeader>
                <CardTitle className="text-base">来源分布</CardTitle>
              </CardHeader>
              <CardContent className="space-y-2">
                {Object.entries(stats.sources ?? {}).map(([name, count]) => (
                  <div key={name} className="flex items-center justify-between rounded-lg border border-border/40 bg-card/40 px-3 py-2 text-sm">
                    <span className="truncate">{name}</span>
                    <span className="ml-3 shrink-0 tabular-nums text-muted-foreground">{count} 块</span>
                  </div>
                ))}
              </CardContent>
            </Card>
          )}
        </div>
      </div>
    </div>
  )
}
