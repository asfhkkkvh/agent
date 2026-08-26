import ReactMarkdown from "react-markdown"
import remarkGfm from "remark-gfm"
import { cn } from "@/lib/utils"

export function Markdown({ content, className }: { content: string; className?: string }) {
  return (
    <div className={cn("max-w-none text-sm leading-relaxed", className)}>
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          p: ({ node: _node, ...props }) => <p className="mb-3" {...props} />,
          a: ({ node: _node, ...props }) => (
            <a {...props} className="text-primary underline underline-offset-4 hover:opacity-80" target="_blank" rel="noreferrer" />
          ),
          code: ({ node: _node, className: cls, children, ...props }) => {
            const isBlock = Boolean(cls?.includes("language-"))
            if (isBlock) {
              return (
                <code className="block overflow-x-auto rounded-lg bg-black/40 px-3 py-2 text-[13px] leading-relaxed" {...props}>
                  {children}
                </code>
              )
            }
            return (
              <code className="rounded bg-muted px-1.5 py-0.5 text-[0.85em] text-primary" {...props}>
                {children}
              </code>
            )
          },
          pre: ({ node: _node, children }) => <pre className="mb-3 overflow-hidden rounded-xl border border-border/60 bg-black/30 p-0">{children}</pre>,
          table: ({ node: _node, ...props }) => (
            <div className="overflow-x-auto rounded-xl border border-border/60">
              <table className="w-full text-left text-sm" {...props} />
            </div>
          ),
          th: ({ node: _node, ...props }) => <th className="border-b border-border/60 bg-muted/50 px-3 py-2 font-semibold" {...props} />,
          td: ({ node: _node, ...props }) => <td className="border-b border-border/40 px-3 py-2 align-top" {...props} />,
          h1: ({ node: _node, ...props }) => <h1 className="mb-2 mt-4 text-xl font-bold" {...props} />,
          h2: ({ node: _node, ...props }) => <h2 className="mb-2 mt-4 text-lg font-bold" {...props} />,
          h3: ({ node: _node, ...props }) => <h3 className="mb-1.5 mt-3 text-base font-semibold" {...props} />,
          ul: ({ node: _node, ...props }) => <ul className="mb-3 list-disc space-y-1 pl-5" {...props} />,
          ol: ({ node: _node, ...props }) => <ol className="mb-3 list-decimal space-y-1 pl-5" {...props} />,
          li: ({ node: _node, ...props }) => <li className="leading-relaxed" {...props} />,
          blockquote: ({ node: _node, ...props }) => (
            <blockquote className="mb-3 border-l-2 border-primary/60 pl-3 text-muted-foreground italic" {...props} />
          ),
          hr: () => <hr className="my-4 border-border/60" />,
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  )
}
