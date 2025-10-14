import { act, fireEvent, render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import App, { OVERLAY_IDLE_HIDE_MS } from "../App";
import type { BackendEvent } from "../services/socket";

const listeners = new Set<(event: BackendEvent) => void>();

vi.mock("../services/socket", () => ({
  connect: vi.fn(),
  onMessage: (listener: (event: BackendEvent) => void) => {
    listeners.add(listener);
    return () => {
      listeners.delete(listener);
    };
  },
  sendQuery: vi.fn(),
  startSession: vi.fn(),
  stopSession: vi.fn(),
  __mockEmit: (event: BackendEvent) => {
    listeners.forEach((listener) => listener(event));
  },
}));

const mockSocket = (await import("../services/socket")) as typeof import("../services/socket") & {
  __mockEmit: (event: BackendEvent) => void;
};

const emit = (event: BackendEvent) => {
  mockSocket.__mockEmit(event);
};

describe("App integration", () => {
  beforeEach(() => {
    listeners.clear();
    localStorage.clear();
  });

  it("renders incoming transcript and aggregates LLM responses", () => {
    render(<App />);

    act(() => {
      emit({ type: "connection", status: "connected" });
      emit({ type: "transcript", id: "t1", text: "Hello world", streaming: false });
    });

    expect(screen.getByText("Hello world")).toBeInTheDocument();

    act(() => {
      emit({ type: "llm", id: "r1", text: "Partial", streaming: true });
    });

    expect(screen.getByText("Cue is responding")).toBeInTheDocument();

    act(() => {
      emit({ type: "llm", id: "r1", text: "Partial answer complete", streaming: false });
    });

    expect(screen.getByText("Partial answer complete")).toBeInTheDocument();
    expect(screen.queryByText("Cue is responding")).not.toBeInTheDocument();
  });

  it("toggles hints visibility", () => {
    render(<App />);

    act(() => {
      emit({ type: "hint", id: "h1", text: "Read the docs", streaming: false });
    });

    expect(screen.getByText("Read the docs")).toBeInTheDocument();

    const toggle = screen.getByRole("button", { name: /disable auto hints/i });
    act(() => {
      fireEvent.click(toggle);
    });

    act(() => {
      emit({ type: "hint", id: "h2", text: "Hidden hint", streaming: false });
    });

    expect(screen.queryByText("Hidden hint")).not.toBeInTheDocument();
    expect(toggle).toHaveAttribute("aria-pressed", "false");
    expect(toggle).toHaveAccessibleName(/enable auto hints/i);
  });

  it("auto hides and reveals the stealth overlay around activity", () => {
    vi.useFakeTimers();
    try {
      render(<App />);

      const overlay = screen.getByTestId("stealth-overlay");
      expect(overlay.className).toContain("opacity-100");

      act(() => {
        vi.advanceTimersByTime(OVERLAY_IDLE_HIDE_MS);
      });

      expect(overlay.className).toContain("opacity-0");

      act(() => {
        emit({ type: "transcript", id: "idle-1", text: "Fresh activity", streaming: false });
      });

      expect(overlay.className).toContain("opacity-100");

      act(() => {
        vi.advanceTimersByTime(OVERLAY_IDLE_HIDE_MS);
      });

      expect(overlay.className).toContain("opacity-0");
    } finally {
      vi.useRealTimers();
    }
  });
});
