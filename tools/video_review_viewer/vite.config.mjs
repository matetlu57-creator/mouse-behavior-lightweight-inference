import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { resolveValidationVideoPath } from "./src/video-paths.js";

const appRoot = path.dirname(fileURLToPath(import.meta.url));
const outputRoot = path.resolve(appRoot, "../../outputs");

function validationVideoServer() {
  function handler(req, res, next) {
    if (!req.url) return next();
    const requested = decodeURIComponent(req.url.split("?", 1)[0]).replace(/^\/+/, "");
    if (!requested.toLowerCase().endsWith(".mp4")) return next();
    const { videoPath, allowed } = resolveValidationVideoPath(requested, outputRoot);
    if (!allowed) {
      res.statusCode = 403;
      res.setHeader("Content-Type", "application/json; charset=utf-8");
      res.end(JSON.stringify({ error: "video path is outside outputs root" }));
      return;
    }
    if (!fs.existsSync(videoPath) || !fs.statSync(videoPath).isFile()) {
      res.statusCode = 404;
      res.setHeader("Content-Type", "application/json; charset=utf-8");
      res.end(JSON.stringify({ error: "rendered video is unavailable", path: requested }));
      return;
    }

    const stat = fs.statSync(videoPath);
    const range = req.headers.range;
    let start = 0;
    let end = stat.size - 1;
    let status = 200;
    if (range) {
      const match = /^bytes=(\d*)-(\d*)$/.exec(range);
      if (match) {
        start = match[1] ? Number(match[1]) : Math.max(0, stat.size - Number(match[2] || 0));
        end = match[2] ? Number(match[2]) : end;
        if (start > end || start >= stat.size) {
          res.statusCode = 416;
          res.setHeader("Content-Range", `bytes */${stat.size}`);
          res.end();
          return;
        }
        end = Math.min(end, stat.size - 1);
        status = 206;
      }
    }

    res.statusCode = status;
    res.setHeader("Content-Type", "video/mp4");
    res.setHeader("Accept-Ranges", "bytes");
    res.setHeader("Content-Length", String(end - start + 1));
    if (status === 206) res.setHeader("Content-Range", `bytes ${start}-${end}/${stat.size}`);
    if (req.method === "HEAD") {
      res.end();
      return;
    }
    fs.createReadStream(videoPath, { start, end }).pipe(res);
  }

  return {
    name: "validation-video-server",
    configureServer(server) {
      server.middlewares.use("/video-files", handler);
    },
    configurePreviewServer(server) {
      server.middlewares.use("/video-files", handler);
    },
  };
}

export default defineConfig({
  build: {
    outDir: "dist/client",
  },
  optimizeDeps: {
    include: ["react", "react-dom/client"],
  },
  server: {
    host: "0.0.0.0",
    allowedHosts: ["terminal.local"],
      fs: { allow: [appRoot, path.resolve(appRoot, "../.."), outputRoot] },
    warmup: {
      clientFiles: ["./src/main.jsx"],
    },
  },
  plugins: [react(), validationVideoServer()],
});
