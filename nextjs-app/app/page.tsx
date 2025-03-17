"use client"

import type React from "react"

import { useState, useRef, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Loader2, Upload } from "lucide-react"
import Image from "next/image"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { Progress } from "@/components/ui/progress"
import { FFmpeg } from "@ffmpeg/ffmpeg"
import { fetchFile, toBlobURL } from "@ffmpeg/util"
import { set } from "date-fns"

export default function VideoUploadForm() {
  const [file, setFile] = useState<File | null>(null)
  const [processedFile, setProcessedFile] = useState<File | null>(null)
  const [sentiment, setSentiment] = useState<string>("")
  const [isLoading, setIsLoading] = useState(false)
  const [isProcessing, setIsProcessing] = useState(false)
  const [processingProgress, setProcessingProgress] = useState(0)
  const [responseImage, setResponseImage] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [ffmpegLoaded, setFfmpegLoaded] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const videoRef = useRef<HTMLVideoElement>(null)
  const ffmpegRef = useRef(new FFmpeg())

  // Load FFmpeg on component mount
  useEffect(() => {
    const loadFFmpeg = async () => {
      try {
        const ffmpeg = ffmpegRef.current

        // Load FFmpeg core
        const baseURL = "https://unpkg.com/@ffmpeg/core@0.12.6/dist/umd"
        await ffmpeg.load({
          coreURL: await toBlobURL(`${baseURL}/ffmpeg-core.js`, "text/javascript"),
          wasmURL: await toBlobURL(`${baseURL}/ffmpeg-core.wasm`, "application/wasm"),
        })

        setFfmpegLoaded(true)
        console.log("FFmpeg loaded successfully")
      } catch (error) {
        console.error("Error loading FFmpeg:", error)
        setError("Failed to load video processing library. Please try again later.")
      }
    }

    loadFFmpeg()

    // Create a hidden video element for getting video duration
    if (!videoRef.current) {
      const video = document.createElement("video")
      video.style.display = "none"
      document.body.appendChild(video)
      videoRef.current = video
    }

    return () => {
      if (videoRef.current && document.body.contains(videoRef.current)) {
        document.body.removeChild(videoRef.current)
      }
    }
  }, [])

  const getVideoDuration = (file: File): Promise<number> => {
    return new Promise((resolve, reject) => {
      if (!videoRef.current) {
        reject(new Error("Video element not available"))
        return
      }

      const video = videoRef.current
      const objectUrl = URL.createObjectURL(file)

      video.onloadedmetadata = () => {
        URL.revokeObjectURL(objectUrl)
        resolve(video.duration)
      }

      video.onerror = () => {
        URL.revokeObjectURL(objectUrl)
        reject(new Error("Error loading video metadata"))
      }

      video.src = objectUrl
    })
  }

  const processVideo = async (inputFile: File) => {
    if (!ffmpegLoaded) {
      setError("Video processing library is not loaded yet. Please wait or refresh the page.")
      return null
    }

    try {
      setIsProcessing(true)
      setProcessingProgress(0)

      // Get video duration
      const duration = await getVideoDuration(inputFile)

      // If video is less than 10 minutes, return the original file
      if (duration <= 600) {
        // 10 minutes = 600 seconds
        console.log("Video is shorter than 10 minutes, using original file")
        setIsProcessing(false)
        setProcessingProgress(100)
        return inputFile
      }

      // Otherwise, trim the video to 10 minutes
      console.log("Trimming video to first 10 minutes")

      const ffmpeg = ffmpegRef.current

      // Set up progress tracking
      ffmpeg.on("progress", ({ progress }) => {
        setProcessingProgress(Math.round(progress * 100))
      })

      // Get the file extension
      const fileExtension = inputFile.name.split(".").pop()?.toLowerCase()
      const inputFileName = `input.${fileExtension}`
      const outputFileName = `output.${fileExtension}`

      // Write the file to FFmpeg's file system
      await ffmpeg.writeFile(inputFileName, await fetchFile(inputFile))

      // Run FFmpeg command to trim the video
      await ffmpeg.exec([
        "-i",
        inputFileName,
        "-t",
        "600", // Limit to 10 minutes
        "-c",
        "copy", // Copy codecs without re-encoding (faster)
        outputFileName,
      ])

      // Read the result
      const data = await ffmpeg.readFile(outputFileName)

      // Create a new File object
      const trimmedFile = new File([new Blob([data], { type: inputFile.type })], `trimmed_${inputFile.name}`, {
        type: inputFile.type,
      })

      // Clean up FFmpeg file system
      await ffmpeg.deleteFile(inputFileName)
      await ffmpeg.deleteFile(outputFileName)

      setIsProcessing(false)
      setProcessingProgress(100)

      return trimmedFile
    } catch (error) {
      console.error("Error processing video:", error)
      setError("Failed to process video. Please try again with a different file.")
      setIsProcessing(false)
      return null
    }
  }

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const selectedFile = e.target.files[0]

      // Check file type
      const fileType = selectedFile.type
      if (fileType !== "video/mp4" && fileType !== "video/webm") {
        setError("Only MP4 and WEBM files are allowed")
        return
      }

      // Check file size (2GB = 2 * 1024 * 1024 * 1024 bytes)
      if (selectedFile.size > 2 * 1024 * 1024 * 1024) {
        setError("File size must be less than 2GB")
        return
      }

      setFile(selectedFile)
      setProcessedFile(null) // Reset processed file
      setError(null)
    }
  }

  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    e.stopPropagation()
  }

  const handleDrop = async (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    e.stopPropagation()

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const selectedFile = e.dataTransfer.files[0]

      // Check file type
      const fileType = selectedFile.type
      if (fileType !== "video/mp4" && fileType !== "video/webm") {
        setError("Only MP4 and WEBM files are allowed")
        return
      }

      // Check file size (2GB = 2 * 1024 * 1024 * 1024 bytes)
      if (selectedFile.size > 2 * 1024 * 1024 * 1024) {
        setError("File size must be less than 2GB")
        return
      }

      setFile(selectedFile)
      setProcessedFile(null) // Reset processed file
      setError(null)
    }
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()

    if (!file) {
      setError("Please select a video file")
      return
    }

    if (!sentiment) {
      setError("Please select a sentiment")
      return
    }

    try {
      // Process the video if not already processed
      const processed = await processVideo(file)
      setProcessedFile(processed)
      if(!processed) {
        throw new Error("Failed to process video")
      }
      console.log("Size after processing: ", processed.size)

      // Now upload the processed file
      setIsLoading(true)
      setError(null)
      setResponseImage(null)

      const formData = new FormData()
      formData.append("video", processed)
      formData.append("sentiment", sentiment)

      const response = await fetch("/image-api/process_video", {
        method: "POST",
        body: formData,
      })

      if (!response.ok) {
        throw new Error("Request failed")
      }

      // Retrieve the job id from the API response.
      const { job_id: jobId } = await response.json();

      // Poll the job status every 10 seconds.
      const pollJobStatus = async () => {
        try {
          const statusRes = await fetch(`/image-api/job_status/${jobId}`);
          if (!statusRes.ok) {
            throw new Error("Job status request failed");
          }
          const jobData = await statusRes.json();
          if (jobData.status === "inprogress") {
            // If still in progress, poll again in 10 seconds.
            setTimeout(pollJobStatus, 10000);
          } else if (jobData.status === "failed") {
            setError(jobData.error || "Job failed");
            setIsLoading(false);
          } else if (jobData.status === "completed") {
            // When completed, create an image URL from the base64 string.
            const imageUrl = `data:image/png;base64,${jobData.image}`;
            setResponseImage(imageUrl);
            setIsLoading(false);
          } else {
            // In case of unexpected status, poll again.
            setTimeout(pollJobStatus, 10000);
          }
        } catch (pollError) {
          setError("Something went wrong while checking job status");
          setIsLoading(false);
          console.error(pollError);
        }
      };

      pollJobStatus();
    } catch (err) {
      setError("Something went wrong")
      console.error(err)
    } finally {
      // setIsLoading(false)
    }
  }

  return (
    <div className="container mx-auto py-10 px-4">
      <Card className="max-w-2xl mx-auto">
        <CardHeader>
          <CardTitle className="text-2xl">Video Sentiment Analysis</CardTitle>
        </CardHeader>
        <CardContent>
          <form onSubmit={handleSubmit} className="space-y-6">
            {/* Video Upload Area */}
            <div
              className="border-2 border-dashed rounded-lg p-6 text-center cursor-pointer hover:bg-muted/50 transition-colors"
              onClick={() => fileInputRef.current?.click()}
              onDragOver={handleDragOver}
              onDrop={handleDrop}
            >
              <input
                type="file"
                ref={fileInputRef}
                onChange={handleFileChange}
                accept="video/mp4,video/webm"
                className="hidden"
              />
              <div className="flex flex-col items-center justify-center gap-2">
                <Upload className="h-10 w-10 text-muted-foreground" />
                <p className="text-sm font-medium">
                  {file ? file.name : "Click to upload or drag and drop a video file"}
                </p>
                <p className="text-xs text-muted-foreground">
                  {file ? `${(file.size / (1024 * 1024)).toFixed(2)} MB` : "MP4, WEBM up to 2GB"}
                </p>
              </div>
            </div>
            <p className="text-xs text-muted-foreground mt-2">
              Note: Only the first 10 minutes of the video will be processed. Videos longer than 10 minutes will be
              trimmed in your browser before uploading.
            </p>

            {/* Processing Progress */}
            {isProcessing && (
              <div className="space-y-2">
                <div className="flex justify-between text-sm">
                  <span>Processing video...</span>
                  <span>{processingProgress}%</span>
                </div>
                <Progress value={processingProgress} className="h-2" />
              </div>
            )}

            {/* Sentiment Dropdown */}
            <div className="space-y-2">
              <label htmlFor="sentiment" className="text-sm font-medium">
                Select Sentiment
              </label>
              <Select value={sentiment} onValueChange={setSentiment}>
                <SelectTrigger id="sentiment">
                  <SelectValue placeholder="Select sentiment" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="positive">Positive</SelectItem>
                  <SelectItem value="negative">Negative</SelectItem>
                  <SelectItem value="neutral">Neutral</SelectItem>
                </SelectContent>
              </Select>
            </div>

            {/* Submit Button */}
            <Button
              type="submit"
              className="w-full"
              disabled={isLoading || isProcessing || !file || !sentiment || !ffmpegLoaded}
            >
              {isLoading ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Processing...
                </>
              ) : (
                "Submit"
              )}
            </Button>

            {/* Error Message */}
            {error && (
              <Alert variant="destructive">
                <AlertDescription>{error}</AlertDescription>
              </Alert>
            )}

            {/* Response Image */}
            {responseImage && (
              <div className="mt-6">
                <h3 className="text-lg font-medium mb-2">Result:</h3>
                <div className="border rounded-lg overflow-hidden">
                  <Image
                    src={responseImage || "/placeholder.svg"}
                    alt="Analysis Result"
                    width={600}
                    height={400}
                    className="w-full h-auto"
                  />
                </div>
              </div>
            )}
          </form>
        </CardContent>
      </Card>
    </div>
  )
}

