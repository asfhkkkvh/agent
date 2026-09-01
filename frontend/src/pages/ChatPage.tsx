import { useEffect, useRef, useState } from "react"
import {
  ArrowUp,
  BookOpen,
  GitMerge,
  Globe,
  Loader2,
  Navigation,
  PenLine,
  Plus,
  RefreshCw,
  ShieldAlert,
  ShieldCheck,
  Sparkles,
  Timer,
} from "lucide-react"
import { cn } from "@/lib/utils"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Card } from "@/components/ui/card"
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Skeleton } from "@/components/ui/skeleton"
import { Markdown } from "@/components/markdown"
import { streamQuery, type AppConfig, type QueryResult, type StreamStep } from "@/lib/api"

const EXAMPLES = [
  "上传报告中的关键发现是什么？",
  "总结最新的 AI 进展",
  "将我的文档信息与当前新闻对比",
  "知识库中关于收入趋势的数据有哪些？",
]

const ROUTE_META: Record<string, { label: string; cls: string }> = {
  rag_agent: { label: "知识库", cls: "border-emerald-500/30 bg-emerald-500/10 text-emerald-400" },
  web_agent: { label: "网络", cls: "border-sky-500/30 bg-sky-500/10 text-sky-400" },
  both: { label: "双源", cls: "border-violet-500/30 bg-violet-500/10 text-violet-400" },
}

const STEP_META: Record<StreamStep, { label: string; icon: typeof Navigation }> = {
  route: { label: "监督者路由", icon: Navigation },
  rag: { label: "知识库检索", icon: BookOpen },
  web: { label: "网络搜索", icon: Globe },
  both: { label: "双路并行", icon: GitMerge },
  synthesis: { label: "综合生成", icon: PenLine },
}

interface StatusStep {
  step: StreamStep
  detail: string
}

interface Turn {
  id: string
  role: "user" | "assistant"
  query?: string
  answer?: string
  result?: QueryResult
  elapsed?: number
  statusSteps: StatusStep[]
  critique?: { passed: boolean; detail: string }
  threadId?: string | null
  error?: string
  streaming: boolean
}

function RouteBadge({ route }: { route?: string }) {
  if (!route) return null
  const meta = ROUTE_META[route] ?? ROUTE_META.both
  return (
    <Badge variant="outline" className={cn("border text-[11px] font-medium", meta.cls)}>
      {meta.label}
    </Badge>
  )
}

function StatusTimeline({ steps, running }: { steps: StatusStep[]; running: boolean }) {
  if (steps.length === 0 && running) {
    return (
      <div className="flex items-center gap-2 py-1 text-sm text-muted-foreground">
        <Loader2 className="h-4 w-4 animate-spin text-primary" />
        监督者正在分析查询…
      </div>
    )
  }
  return (
    <div className="mb-3 space-y-1.5 rounded-xl border border-border/50 bg-card/50 p-3">
      {steps.map((s, i) => {
        const meta = STEP_META[s.step]
        const Icon = meta.icon
        const isLast = i === steps.length - 1
        return (
          <div key={`${s.step}-${i}`} className="flex items-center gap-2.5 text-[13px]">
            <span
              className={cn(
                "flex h-6 w-6 shrink-0 items-center justify-center rounded-full border",
                isLast && running
                  ? "border-primary/40 bg-primary/10 text-primary"
                  : "border-border bg-muted/60 text-muted-foreground",
              )}
            >
              {isLast && running ? (
                <Loader2 className="h-3.5 w-3.5 animate-spin" />
              ) : (
                <Icon className="h-3.5 w-3.5" />
              )}
            </span>
            <span className={cn("font-medium", isLast && running ? "text-foreground" : "text-muted-foreground")}>
              {meta.label}
            </span>
            {s.detail && <span className="truncate text-xs text-muted-foreground/80">{s.detail}</span>}
          </div>
        )
      })}
    </div>
  )
}

function SourcePanel({ title, icon: Icon, content }: { title: string; icon: typeof BookOpen; content?: string }) {
  if (!content || !content.trim()) return null
  return (
    <Collapsible className="mt-2">
      <CollapsibleTrigger className="group flex w-full items-center gap-2 rounded-lg border border-border/60 bg-card/40 px-3 py-2 text-left text-xs text-muted-foreground transition-colors hover:text-foreground">
        <Icon className="h-3.5 w-3.5 text-primary/80" />
        {title}
        <span className="ml-auto text-primary/70 transition-transform group-data-[state=open]:rotate-180">▾</span>
      </CollapsibleTrigger>
      <CollapsibleContent className="mt-1.5">
        <div className="max-h-64 overflow-auto rounded-lg border border-border/40 bg-black/25 p-3 text-xs leading-relaxed whitespace-pre-wrap text-muted-foreground">
          {content}
        </div>
      </CollapsibleContent>
    </Collapsible>
  )
}

function AssistantCard({ turn }: { turn: Turn }) {
  const { result, streaming } = turn
  return (
    <div className="flex gap-3">
      <div className="mt-0.5 flex h-8 w-8 shrink-0 items-center justify-center rounded-xl bg-gradient-to-br from-primary to-cyan-400 shadow-md shadow-primary/25">
        <Sparkles className="h-4 w-4 text-white" />
      </div>
      <Card className="min-w-0 flex-1 border-border/60 bg-card/70 p-4 shadow-lg shadow-black/10 backdrop-blur-sm">
        {streaming && !turn.answer && (
          <>
            <StatusTimeline steps={turn.statusSteps} running />
            <Skeleton className="h-3 w-full" />
            <Skeleton className="mt-2 h-3 w-4/5" />
            <Skeleton className="mt-2 h-3 w-3/5" />
          </>
        )}

        {turn.error && <p className="text-sm text-red-400">⚠️ {turn.error}</p>}

        {turn.answer && (
          <>
            <div className="mb-2 flex flex-wrap items-center gap-2">
              <RouteBadge route={result?.route} />
              {result?.iterations ? (
                <Badge variant="outline" className="border-border/60 text-[11px] text-muted-foreground">
                  <RefreshCw className="mr-1 h-3 w-3" />
                  {result.iterations} 轮评审
                </Badge>
              ) : null}
              {turn.elapsed !== undefined && (
                <Badge variant="outline" className="border-border/60 text-[11px] text-muted-foreground">
                  <Timer className="mr-1 h-3 w-3" />
                  {turn.elapsed.toFixed(1)}s
                </Badge>
              )}
              {turn.critique?.passed && (
                <Badge variant="outline" className="border-emerald-500/30 bg-emerald-500/10 text-[11px] text-emerald-400">
                  <ShieldCheck className="mr-1 h-3 w-3" />
                  评审通过
                </Badge>
              )}
            </div>

            <Markdown content={turn.answer} />

            {turn.critique && !turn.critique.passed && (
              <div className="mt-3 flex items-start gap-2 rounded-lg border border-amber-500/25 bg-amber-500/5 p-3 text-xs text-amber-300">
                <ShieldAlert className="mt-0.5 h-4 w-4 shrink-0" />
                <div>
                  <div className="font-medium">评审要求修订，已携带反馈重试</div>
                  <div className="mt-1 text-amber-200/70">{turn.critique.detail}</div>
                </div>
              </div>
            )}

            <SourcePanel title="知识库检索上下文" icon={BookOpen} content={result?.rag_context} />
            <SourcePanel title="网络搜索上下文" icon={Globe} content={result?.web_context} />
          </>
        )}
      </Card>
    </div>
  )
}

export function ChatPage({ config }: { config: AppConfig | null }) {
  const [turns, setTurns] = useState<Turn[]>([])
  const [input, setInput] = useState("")
  const [streaming, setStreaming] = useState(false)
  const threadIdRef = useRef<string | null>(null)
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [turns])

  function newConversation() {
    threadIdRef.current = null
    setTurns([])
  }

  async function handleSend(text: string) {
    const query = text.trim()
    if (!query || streaming) return
    setInput("")

    const assistantId = crypto.randomUUID()
    const userTurn: Turn = { id: crypto.randomUUID(), role: "user", query, statusSteps: [], streaming: false }
    const assistantTurn: Turn = { id: assistantId, role: "assistant", statusSteps: [], streaming: true }
    setTurns((prev) => [...prev, userTurn, assistantTurn])
    setStreaming(true)

    const started = performance.now()
    try {
      await streamQuery(query, threadIdRef.current, (ev) => {
        if (ev.type === "start" && ev.thread_id) {
          threadIdRef.current = ev.thread_id
        } else if (ev.type === "status" && ev.step && ev.detail) {
          const step: StatusStep = { step: ev.step, detail: ev.detail }
          setTurns((prev) =>
            prev.map((t) => (t.id === assistantId ? { ...t, statusSteps: [...t.statusSteps, step] } : t)),
          )
        } else if (ev.type === "critique") {
          setTurns((prev) =>
            prev.map((t) =>
              t.id === assistantId ? { ...t, critique: { passed: !!ev.passed, detail: ev.detail ?? "" } } : t,
            ),
          )
        } else if (ev.type === "final" && ev.result) {
          setTurns((prev) =>
            prev.map((t) => (t.id === assistantId ? { ...t, result: ev.result, answer: ev.result!.final_answer } : t)),
          )
        } else if (ev.type === "error") {
          setTurns((prev) =>
            prev.map((t) => (t.id === assistantId ? { ...t, error: ev.detail ?? "未知错误", streaming: false } : t)),
          )
        }
      })
    } catch (e) {
      setTurns((prev) =>
        prev.map((t) => (t.id === assistantId ? { ...t, error: String(e), streaming: false } : t)),
      )
    } finally {
      setTurns((prev) =>
        prev.map((t) =>
          t.id === assistantId ? { ...t, streaming: false, elapsed: (performance.now() - started) / 1000 } : t,
        ),
      )
      setStreaming(false)
    }
  }

  return (
    <div className="flex h-full flex-col">
      {/* 顶部栏 */}
      <header className="flex items-center justify-between border-b border-border/50 px-6 py-4 backdrop-blur-sm">
        <div>
          <h1 className="text-lg font-semibold tracking-tight">智能对话</h1>
          <p className="text-xs text-muted-foreground">
            监督者路由 → 混合检索 → 综合 → 评审修订,全程 LangSmith 可观测
          </p>
        </div>
        <Button variant="outline" size="sm" onClick={newConversation}>
          <Plus className="h-4 w-4" />
          新会话
        </Button>
      </header>

      {/* 消息区 */}
      <ScrollArea className="min-h-0 flex-1">
        <div className="mx-auto max-w-3xl space-y-5 px-6 py-6">
          {turns.length === 0 && (
            <div className="flex flex-col items-center justify-center py-24 text-center">
              <div className="mb-5 flex h-14 w-14 items-center justify-center rounded-3xl bg-gradient-to-br from-primary via-indigo-500 to-cyan-400 shadow-xl shadow-primary/30">
                <Sparkles className="h-7 w-7 text-white" />
              </div>
              <h2 className="text-xl font-semibold">OmniRAG 混合研究助手</h2>
              <p className="mt-2 max-w-md text-sm text-muted-foreground">
                支持文档知识库 + 实时网络双源检索,答案自动带来源引用,并经评审智能体迭代修订。
              </p>
              <div className="mt-7 grid w-full max-w-xl grid-cols-1 gap-2 sm:grid-cols-2">
                {EXAMPLES.map((q) => (
                  <button
                    key={q}
                    onClick={() => handleSend(q)}
                    className="rounded-xl border border-border/60 bg-card/50 px-4 py-3 text-left text-[13px] text-muted-foreground transition-all hover:border-primary/40 hover:bg-card hover:text-foreground"
                  >
                    {q}
                  </button>
                ))}
              </div>
            </div>
          )}

          {turns.map((turn) =>
            turn.role === "user" ? (
              <div key={turn.id} className="flex justify-end">
                <div className="max-w-[80%] rounded-2xl rounded-br-md bg-primary/15 px-4 py-3 text-sm leading-relaxed">
                  {turn.query}
                </div>
              </div>
            ) : (
              <AssistantCard key={turn.id} turn={turn} />
            ),
          )}
          <div ref={bottomRef} />
        </div>
      </ScrollArea>

      {/* 输入区 */}
      <div className="border-t border-border/50 px-6 py-4 backdrop-blur-sm">
        <div className="mx-auto flex max-w-3xl items-end gap-2 rounded-2xl border border-border/60 bg-card/80 p-2 shadow-xl shadow-black/10 backdrop-blur-sm focus-within:border-primary/50">
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault()
                void handleSend(input)
              }
            }}
            rows={1}
            placeholder="输入问题,Enter 发送,Shift+Enter 换行…"
            className="max-h-32 min-h-10 flex-1 resize-none bg-transparent px-3 py-2 text-sm outline-none placeholder:text-muted-foreground/60"
          />
          <Button
            size="icon"
            onClick={() => void handleSend(input)}
            disabled={streaming || !input.trim()}
            className="h-10 w-10 shrink-0 rounded-xl bg-gradient-to-br from-primary to-cyan-500 shadow-md shadow-primary/30"
            aria-label="发送"
          >
            {streaming ? <Loader2 className="h-4 w-4 animate-spin" /> : <ArrowUp className="h-4 w-4" />}
          </Button>
        </div>
        <p className="mx-auto mt-2 max-w-3xl text-center text-[11px] text-muted-foreground/60">
          {config ? `模型 ${config.model} · 检索 Top-${config.retrieval_top_k} → 重排 Top-${config.reranker_top_n}` : ""}
          {config?.use_multi_query ? " · 多查询扩展已开启" : ""}
          {" · 答案由 AI 生成,请核验关键信息"}
        </p>
      </div>
    </div>
  )
}
