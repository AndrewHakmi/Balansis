import { useEffect, useRef } from 'react';

interface MathRendererProps {
  math: string;
  display?: boolean;
  className?: string;
}

declare global {
  interface Window {
    MathJax: {
      typesetPromise: (elements?: Element[]) => Promise<void>;
      startup: {
        promise: Promise<void>;
      };
    };
  }
}

export function MathRenderer({ math, display = false, className = '' }: MathRendererProps) {
  const mathRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const loadMathJax = async () => {
      // Load MathJax if not already loaded
      if (!window.MathJax) {
        // Configure MathJax
        (window as any).MathJax = {
          tex: {
            inlineMath: [['$', '$'], ['\\(', '\\)']],
            displayMath: [['$$', '$$'], ['\\[', '\\]']],
            processEscapes: true,
            processEnvironments: true,
            tags: 'ams',
            macros: {
              // ACT-specific macros
              abs: ['\\left|#1\\right|', 1],
              comp: ['\\text{comp}(#1)', 1],
              eternal: ['\\mathcal{E}(#1)', 1],
              balansis: '\\mathcal{B}',
              act: '\\text{ACT}',
            }
          },
          svg: {
            fontCache: 'global'
          },
          startup: {
            ready: () => {
              const mathJax = (window as any).MathJax;
              if (mathJax && mathJax.startup && mathJax.startup.defaultReady) {
                mathJax.startup.defaultReady();
              }
            }
          }
        };

        // Load MathJax script
        const script = document.createElement('script');
        script.src = 'https://polyfill.io/v3/polyfill.min.js?features=es6';
        document.head.appendChild(script);

        const mathJaxScript = document.createElement('script');
        mathJaxScript.id = 'MathJax-script';
        mathJaxScript.async = true;
        mathJaxScript.src = 'https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js';
        document.head.appendChild(mathJaxScript);

        // Wait for MathJax to load
        await new Promise((resolve) => {
          mathJaxScript.onload = resolve;
        });
      }

      // Wait for MathJax to be ready
      await window.MathJax.startup.promise;

      // Render the math
      if (mathRef.current) {
        mathRef.current.innerHTML = display ? `$$${math}$$` : `$${math}$`;
        await window.MathJax.typesetPromise([mathRef.current]);
      }
    };

    loadMathJax().catch(console.error);
  }, [math, display]);

  return (
    <div 
      ref={mathRef} 
      className={`math-renderer ${display ? 'display-math' : 'inline-math'} ${className}`}
      style={{
        display: display ? 'block' : 'inline-block',
        textAlign: display ? 'center' : 'inherit',
        margin: display ? '1rem 0' : '0',
      }}
    />
  );
}

// Utility component for LaTeX blocks
export function MathBlock({ children, className = '' }: { children: string; className?: string }) {
  return <MathRenderer math={children} display={true} className={className} />;
}

// Utility component for inline LaTeX
export function MathInline({ children, className = '' }: { children: string; className?: string }) {
  return <MathRenderer math={children} display={false} className={className} />;
}