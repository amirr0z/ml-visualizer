
"use client";

import { ThemeProvider as NextThemeProvider } from "next-themes";

export default function ThemeProvider({
    children,
}: {
    children: React.ReactNode;
}) {
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
