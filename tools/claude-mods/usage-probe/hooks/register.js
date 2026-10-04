// Writes the plan's rate-limit windows to the file named by THAI_SYLLABUS_USAGE_OUT
// once a turn completes; rateLimits is empty before the first response.
export function register(on) {
  on('turn.complete', async ($, e, next) => {
    const result = await next(e)
    const path = await $.env.get('THAI_SYLLABUS_USAGE_OUT')
    if (path) {
      const { rateLimits, context, cost } = await $.session.usage()
      await $.fs.write(path, JSON.stringify({ read_at: new Date().toISOString(), rateLimits, context, cost }, null, 2) + '\n')
    }
    return result
  })
}
