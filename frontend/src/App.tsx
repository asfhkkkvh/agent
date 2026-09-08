import { useEffect, useState } from "react"
import { ThemeProvider } from "next-themes"
import { Toaster } from "@/components/ui/sonner"
import { Sidebar } from "@/components/layout/sidebar"
import { ChatPage } from "@/pages/ChatPage"
import { DataPage } from "@/pages/DataPage"
import { EvalPage } from "@/pages/EvalPage"
import { getConfig, type AppConfig } from "@/lib/api"

export type Page = "chat" | "data" | "eval"

function BackgroundFX() {
  return (
    <div aria-hidden className="pointer-events-none fixed inset-0 z-0 overflow-hidden">
      <div className="absolute -top-44 left-1/4 h-[520px] w-[520px] rounded-full bg-primary/14 blur-[150px]" />
      <div className="absolute -bottom-48 right-[18%] h-[460px] w-[460px] rounded-full bg-cyan-500/10 blur-[150px]" />
      <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_top,transparent_0%,transparent_65%,rgba(0,0,0,0.45)_100%)]" />
    </div>
  )
}

function Shell() {
  const [page, setPage] = useState<Page>("chat")
  const [config, setConfig] = useState<AppConfig | null>(null)
  const [apiOk, setApiOk] = useState<boolean | null>(null)

  useEffect(() => {
    getConfig()
      .then((c) => {
        setConfig(c)
        setApiOk(true)
      })
      .catch(() => setApiOk(false))
  }, [])

  return (
    <div className="relative flex h-screen overflow-hidden bg-background text-foreground">
      <BackgroundFX />
      <Sidebar page={page} onNavigate={setPage} config={config} apiOk={apiOk} />
      <main className="relative z-10 min-w-0 flex-1 overflow-hidden">
        {/* 用 CSS display 切换而非条件渲染，避免切页面时组件卸载导致
            SSE 连接断开和对话状态丢失（对话中跳去评估再回来不会中断） */}
        <div style={{ display: page === "chat" ? "flex" : "none" }} className="h-full flex-col">
          <ChatPage config={config} />
        </div>
        <div style={{ display: page === "data" ? "block" : "none" }} className="h-full">
          <DataPage />
        </div>
        <div style={{ display: page === "eval" ? "block" : "none" }} className="h-full">
          <EvalPage />
        </div>
      </main>
    </div>
  )
}

export default function App() {
  return (
    <ThemeProvider attribute="class" defaultTheme="dark" enableSystem={false}>
      <Shell />
      <Toaster position="top-center" richColors />
    </ThemeProvider>
  )
}
