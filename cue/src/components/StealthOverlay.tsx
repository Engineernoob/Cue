import { PropsWithChildren } from "react";

const FADE_CLASS_VISIBLE = "opacity-100";
const FADE_CLASS_HIDDEN = "opacity-0 pointer-events-none";

type StealthOverlayProps = PropsWithChildren<{
  visible: boolean;
}>;

export default function StealthOverlay({ visible, children }: StealthOverlayProps) {
  const classes = [
    "fixed inset-0 z-40 flex h-screen w-screen flex-col items-center justify-center",
    "bg-[radial-gradient(circle_at_top,_rgba(46,54,117,0.72),_rgba(10,12,26,0.92))]",
    "transition-opacity duration-500 ease-out",
    visible ? FADE_CLASS_VISIBLE : FADE_CLASS_HIDDEN,
  ].join(" ");

  return (
    <div data-testid="stealth-overlay" className={classes}>
      <div
        aria-hidden="true"
        className="absolute inset-0 backdrop-blur-3xl backdrop-brightness-[0.85] transition-opacity duration-500"
      />
      <div className="relative z-10 flex h-full w-full max-w-5xl flex-col gap-4 px-6 py-6 text-white">
        {children}
      </div>
    </div>
  );
}
