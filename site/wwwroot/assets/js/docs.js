document.addEventListener("DOMContentLoaded", () => {

    function bindEvents() {
        document.querySelectorAll(".doc-link").forEach(link => {
            link.addEventListener("click", evt => {
                evt.preventDefault();
                const doc = evt.currentTarget.dataset.doc;           
                history.pushState({}, "", evt.currentTarget.href);    
                loadDoc(doc);
            });
        });
    }

    function renderEnhancements() {
        // 1) Converte <pre><code class="language-mermaid">...</code></pre> em <pre class="mermaid">...</pre>
        document.querySelectorAll("#doc-content pre > code.language-mermaid").forEach(codeEl => {
            const pre = codeEl.parentElement; // <pre>
            const container = document.createElement("pre");
            container.className = "mermaid";
            container.textContent = codeEl.textContent;
            pre.replaceWith(container);
        });

        // 2) Destaca os demais códigos
        if (window.hljs) {
            window.hljs.highlightAll();
        }

        // 3) Renderiza diagramas Mermaid
        if (window.mermaid) {
            // se `run` existir (v10+)
            if (typeof window.mermaid.run === "function") {
                window.mermaid.run({
                    nodes: document.querySelectorAll("#doc-content pre.mermaid")
                });
            } else if (typeof window.mermaid.init === "function") {
                window.mermaid.init(undefined, document.querySelectorAll("#doc-content pre.mermaid"));
            }
        }
    }

    function loadDoc(docName) {
        fetch(`/Docs/Content?id=${encodeURIComponent(docName)}`) 
            .then(r => r.ok ? r.text() : Promise.reject(r))
            .then(html => {
                document.getElementById("doc-content").innerHTML = html;
                // atualiza estado visual
                document.querySelectorAll("#doc-menu .active")
                    .forEach(li => li.classList.remove("active"));
                
                // Usa uma abordagem mais robusta para encontrar o elemento
                const menuItem = document.querySelector(`#doc-menu [data-doc="${CSS.escape(docName)}"]`);
                if (menuItem) {
                    menuItem.parentElement.classList.add("active");
                } else {
                    console.warn("Item do menu não encontrado para:", docName);
                }

                renderEnhancements();
            })
            .catch(err => console.error("Erro carregando doc:", err));
    }

    bindEvents();

    // Suporte ao botão "voltar"
    window.addEventListener("popstate", () => {
        const id = new URL(location).searchParams.get("id");
        if (id) loadDoc(id);
    });

    // garante mermaid na primeira renderização
    window.addEventListener("load", () => {
        renderEnhancements();
    });
});
