export default {
  async fetch(request, env) {
    const response = await env.ASSETS.fetch(request);
    const acceptsHtml = request.headers.get("accept")?.includes("text/html");

    const pathname = new URL(request.url).pathname;
    if (response.status !== 404 || !acceptsHtml || !["GET", "HEAD"].includes(request.method)
        || /\.(json|mp4)$/i.test(pathname)) {
      return response;
    }

    const indexUrl = new URL(request.url);
    indexUrl.pathname = "/index.html";
    indexUrl.search = "";
    return env.ASSETS.fetch(new Request(indexUrl, request));
  },
};
