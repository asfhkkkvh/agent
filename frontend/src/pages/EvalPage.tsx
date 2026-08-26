import { useState } from "react"
import { toast } from "sonner"
import { Activity, CheckCircle2, FileBarChart2, FlaskConical, Loader2, Play } from "lucide-react"
import { cn } from "@/lib/utils"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Label } from "@/components/ui/label"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Separator } from "@/components/ui/separator"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { runEval, type EvalResult } from "@/lib/api"

const METRICS: { key: string; label: string; desc: string; tint: string }[] = [
  { key: "faithfulness", label: "Faithfulness", desc: "答案忠于上下文,无幻觉", tint: "bg-emerald-500" },
  { key: "answer_relevancy", label: "Answer Relevancy", desc: "答案切题程度", tint: "bg-primary" },
  { key: "context_precision", label: "Context Precision", desc: "检索上下文精炼度", tint: "bg-violet-500" },
  { key: "context_recall", label: "Context Recall", desc: "检索上下文召回率", tint: "bg-cyan-500" },
]

function MetricCard({ label, desc, value, tint }: { label: string; desc: string; value?: number; tint: string }) {
  const pct = value !== undefined ? Math.round(value * 100) : null
  return (
    <Card className="border-border/60 bg-card/60 backdrop-blur-sm">
      <CardContent className="p-5">
        <div className="flex items-start justify-between">
          <div>
            <div className="text-sm font-semibold">{label}</div>
            <div className="mt-0.5 text-xs text-muted-foreground">{desc}</div>
          </div>
          <Activity className="h-4 w-4 text-muted-foreground/60" />
        </div>
        <div className="mt-4 text-3xl font-semibold tabular-nums">{pct === null ? "—" : `${pct}%`}</div>
        <div className="mt-3 h-1.5 overflow-hidden rounded-full bg-muted">
          <div
            className={cn("h-full rounded-full transition-all duration-700", tint)}
            style={{ width: pct === null ? "0%" : `${pct}%` }}
          />
        </div>
      </CardContent>
    </Card>
  )
}

export function EvalPage() {
  const [sampleCount, setSampleCount] = useState("5")
  const [running, setRunning] = useState(false)
  const [result, setResult] = useState<EvalResult | null>(null)
  const [error, setError] = useState<string | null>(null)

  async function handleRun() {
    setRunning(true)
    setError(null)
    setResult(null)
    const started = performance.now()
    try {
      const res = await runEval(Number(sampleCount))
      if (res.error) {
        setError(res.error)
      } else {
        setResult(res)
        toast.success(`评估完成,耗时 ${((performance.now() - started) / 1000).toFixed(1)}s`)
      }
    } catch (e) {
      setError(String(e))
    } finally {
      setRunning(false)
    }
  }

  return (
    <div className="flex h-full flex-col overflow-hidden">
      <header className="border-b border-border/50 px-6 py-4 backdrop-blur-sm">
        <h1 className="text-lg font-semibold tracking-tight">RAGAS 质量评估</h1>
        <p className="text-xs text-muted-foreground">
          自动生成评测样本 → 完整管道推理 → RAGAS 四维指标量化,报告自动存档至 data/eval_reports
        </p>
      </header>

      <div className="min-h-0 flex-1 overflow-y-auto">
        <div className="mx-auto max-w-4xl space-y-6 px-6 py-6">
          {/* 控制区 */}
          <Card className="border-border/60 bg-card/50 backdrop-blur-sm">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-base">
                <FlaskConical className="h-4 w-4 text-primary" />
                运行评估
              </CardTitle>
              <CardDescription>评测会调用完整多 Agent 管道与 RAGAS judge,样本越多耗时越长</CardDescription>
            </CardHeader>
            <CardContent className="flex flex-wrap items-end gap-3">
              <div className="space-y-1.5">
                <Label>样本数量</Label>
                <Select value={sampleCount} onValueChange={setSampleCount}>
                  <SelectTrigger className="w-32">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {["3", "5", "10", "20"].map((n) => (
                      <SelectItem key={n} value={n}>
                        {n} 条
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              <Button onClick={() => void handleRun()} disabled={running} className="gap-2">
                {running ? <Loader2 className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4" />}
                {running ? "评估中…" : "开始评估"}
              </Button>
              {result?.report_path && (
                <span className="flex items-center gap-1.5 text-xs text-muted-foreground">
                  <FileBarChart2 className="h-3.5 w-3.5 text-emerald-400" />
                  报告: {result.report_path}
                </span>
              )}
            </CardContent>
          </Card>

          {error && (
            <p className="rounded-xl border border-red-500/30 bg-red-500/10 px-4 py-3 text-sm text-red-300">{error}</p>
          )}

          {/* 指标 */}
          {result && (
            <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
              {METRICS.map((m) => (
                <MetricCard key={m.key} label={m.label} desc={m.desc} value={result.metrics[m.key]} tint={m.tint} />
              ))}
            </div>
          )}

          {/* 样本明细 */}
          {result && result.samples && result.samples.length > 0 && (
            <Card className="border-border/60 bg-card/50 backdrop-blur-sm">
              <CardHeader>
                <CardTitle className="flex items-center gap-2 text-base">
                  <CheckCircle2 className="h-4 w-4 text-emerald-400" />
                  样本明细（{result.samples.length} 条）
                </CardTitle>
              </CardHeader>
              <CardContent>
                <Table>
                  <TableHeader>
                    <TableRow className="hover:bg-transparent">
                      <TableHead className="w-1/3">问题</TableHead>
                      <TableHead className="w-1/3">标准答案</TableHead>
                      <TableHead>生成答案</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {result.samples.map((s, i) => (
                      <TableRow key={i} className="align-top">
                        <TableCell className="text-xs leading-relaxed">{s.query}</TableCell>
                        <TableCell className="text-xs leading-relaxed text-muted-foreground">{s.ground_truth}</TableCell>
                        <TableCell className="text-xs leading-relaxed text-muted-foreground">{s.answer}</TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </CardContent>
            </Card>
          )}

          {result && <Separator />}
        </div>
      </div>
    </div>
  )
}
