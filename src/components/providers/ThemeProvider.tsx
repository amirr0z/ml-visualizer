
"use client";

import { ThemeProvider as NextThemeProvider } from "next-themes";
import type { ReactNode } from "react";

interface Props {
    children: ReactNode;
}

export default function ThemeProvider({ children }: Props) {
    return (
        <NextThemeProvider
            attribute="data-theme"
            defaultTheme="dark"
            enableSystem={false}
            storageKey="ml-laboratory-theme"
            disableTransitionOnChange={false}
        >
            {children}
        </NextThemeProvider>
    );
}
