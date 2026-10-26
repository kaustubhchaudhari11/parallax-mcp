(function () {
  const moved = {
    findings: "research/#findings",
    question: "research/#question",
    limits: "research/#limits",
    method: "how-it-works/",
    byoa: "try/",
    try: "try/"
  };
  if (document.body.dataset.page === "home" && window.location.hash) {
    const id = window.location.hash.slice(1);
    if (moved[id] && !document.getElementById(id)) {
      window.location.replace(moved[id]);
    }
  }

  const toggle = document.querySelector(".nav-toggle");
  const links = document.querySelector(".nav-links");
  if (toggle && links) {
    toggle.addEventListener("click", function () {
      const open = links.classList.toggle("is-open");
      toggle.setAttribute("aria-expanded", open ? "true" : "false");
    });
    links.querySelectorAll("a").forEach(function (link) {
      link.addEventListener("click", function () {
        links.classList.remove("is-open");
        toggle.setAttribute("aria-expanded", "false");
      });
    });
  }

  document.querySelectorAll("[data-instrument]").forEach(function (instrument) {
    const radios = Array.prototype.slice.call(
      instrument.querySelectorAll('input[type="radio"]')
    );
    const readings = instrument.querySelectorAll("[data-reading]");
    const apply = function () {
      const selected = instrument.querySelector('input[type="radio"]:checked');
      const value = selected ? selected.value : "";
      radios.forEach(function (radio) {
        const node = radio.closest(".node");
        if (node) {
          node.classList.toggle("is-selected", radio.checked);
        }
      });
      readings.forEach(function (reading) {
        reading.classList.toggle("is-selected", reading.getAttribute("data-reading") === value);
      });
      instrument.classList.add("is-ready");
    };
    radios.forEach(function (radio) {
      radio.addEventListener("change", apply);
      radio.addEventListener("keydown", function (event) {
        const index = radios.indexOf(radio);
        let next = null;
        if (event.key === "ArrowRight" || event.key === "ArrowDown") {
          next = (index + 1) % radios.length;
        } else if (event.key === "ArrowLeft" || event.key === "ArrowUp") {
          next = (index - 1 + radios.length) % radios.length;
        } else if (event.key === "Home") {
          next = 0;
        } else if (event.key === "End") {
          next = radios.length - 1;
        }
        if (next === null) {
          return;
        }
        event.preventDefault();
        radios[next].checked = true;
        radios[next].focus();
        radios[next].dispatchEvent(new Event("change", { bubbles: true }));
      });
    });
    apply();
  });

  document.querySelectorAll("[data-tabs]").forEach(function (widget) {
    const tabs = Array.prototype.slice.call(widget.querySelectorAll('[role="tab"]'));
    const panels = widget.querySelectorAll('[role="tabpanel"]');
    const select = function (index, focusTab) {
      tabs.forEach(function (tab, tabIndex) {
        const active = tabIndex === index;
        tab.setAttribute("aria-selected", active ? "true" : "false");
        tab.tabIndex = active ? 0 : -1;
        if (active && focusTab) {
          tab.focus();
        }
      });
      panels.forEach(function (panel) {
        panel.classList.toggle("is-selected", panel.id === tabs[index].getAttribute("aria-controls"));
      });
      widget.classList.add("is-ready");
    };
    tabs.forEach(function (tab, index) {
      tab.addEventListener("click", function () {
        select(index, false);
      });
      tab.addEventListener("keydown", function (event) {
        let next = null;
        if (event.key === "ArrowRight") {
          next = (index + 1) % tabs.length;
        } else if (event.key === "ArrowLeft") {
          next = (index - 1 + tabs.length) % tabs.length;
        } else if (event.key === "Home") {
          next = 0;
        } else if (event.key === "End") {
          next = tabs.length - 1;
        }
        if (next === null) {
          return;
        }
        event.preventDefault();
        select(next, true);
      });
    });
    widget.classList.add("is-ready");
  });

  const status = document.querySelector(".copy-status");
  document.querySelectorAll("[data-copy]").forEach(function (button) {
    button.addEventListener("click", function () {
      const target = document.getElementById(button.getAttribute("data-copy"));
      const text = target ? target.textContent.replace(/\n$/, "") : "";
      const snippet = button.closest(".snippet");
      writeClipboard(text).then(
        function () {
          button.textContent = "Copied";
          if (status) {
            status.textContent = "Copied";
          }
          clearNote(snippet);
          window.setTimeout(function () {
            if (button.textContent === "Copied") {
              button.textContent = "Copy";
            }
            if (status && status.textContent === "Copied") {
              status.textContent = "";
            }
          }, 1600);
        },
        function () {
          showCopyFallback(button, target, snippet);
        }
      );
    });
  });

  function clearNote(snippet) {
    if (!snippet) {
      return;
    }
    const note = snippet.querySelector(".copy-note");
    if (note) {
      note.hidden = true;
      note.textContent = "";
    }
  }

  function showCopyFallback(button, target, snippet) {
    let message = "Automatic copy did not work. Select the text in the panel and press Ctrl+C or Command+C.";
    if (target) {
      try {
        const range = document.createRange();
        range.selectNodeContents(target);
        const selection = window.getSelection();
        if (selection) {
          selection.removeAllRanges();
          selection.addRange(range);
          message = "Automatic copy did not work. The text is selected. Press Ctrl+C or Command+C.";
        }
      } catch (error) {
        message = "Automatic copy did not work. Select the text in the panel and press Ctrl+C or Command+C.";
      }
    }
    button.textContent = "Select and copy";
    if (status) {
      status.textContent = message;
    }
    if (!snippet) {
      return;
    }
    let note = snippet.querySelector(".copy-note");
    if (!note) {
      note = document.createElement("p");
      note.className = "copy-note";
      snippet.appendChild(note);
    }
    note.hidden = false;
    note.textContent = message;
  }

  function writeClipboard(text) {
    const legacy = function () {
      return new Promise(function (resolve, reject) {
        const area = document.createElement("textarea");
        area.value = text;
        area.setAttribute("readonly", "");
        area.style.position = "fixed";
        area.style.left = "-9999px";
        document.body.appendChild(area);
        area.select();
        const ok = document.execCommand("copy");
        area.remove();
        if (ok) {
          resolve();
        } else {
          reject(new Error("copy failed"));
        }
      });
    };
    if (navigator.clipboard && window.isSecureContext) {
      return navigator.clipboard.writeText(text).catch(legacy);
    }
    return legacy();
  }
})();
